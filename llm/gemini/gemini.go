package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/henomis/lingoose/llm/cache"
	"github.com/henomis/lingoose/thread"
	"golang.org/x/oauth2/google"
	"google.golang.org/genai"
	"strings"
)

const (
	EOS = "\x00"
)

var threadRoleToGeminiRole = map[thread.Role]string{
	thread.RoleSystem:    "system_instruction",
	thread.RoleUser:      "user",
	thread.RoleAssistant: "model",
	thread.RoleTool:      "tool",
}

type Gemini struct {
	ctx              context.Context
	client           *genai.Client
	generateConfig   *genai.GenerateContentConfig
	model            Model
	temperature      float32
	maxTokens        int
	stop             []string
	functions        map[string]Function
	streamCallbackFn StreamCallback
	tools            []*genai.Tool
	cache            *cache.Cache
	TotalTokens      int64
}

func DefaultSafetySettings() []*genai.SafetySetting {
	return []*genai.SafetySetting{{
		Category:  genai.HarmCategoryDangerousContent,
		Threshold: genai.HarmBlockThresholdBlockMediumAndAbove,
	}}
}

// WithTemperature sets the temperature to use for the Gemini instance.
func (g *Gemini) WithTemperature(temperature float32) *Gemini {
	g.temperature = temperature
	return g
}

func (g *Gemini) WithStream(enable bool, callbackFn StreamCallback) *Gemini {
	if !enable {
		g.streamCallbackFn = nil
	} else {
		g.streamCallbackFn = callbackFn
	}
	return g
}

func (g *Gemini) ClearTools() {
	g.generateConfig.Tools = []*genai.Tool{}
}

func (g *Gemini) WithTools(tools []*genai.Tool) *Gemini {
	g.generateConfig.Tools = tools
	return g
}

func (g *Gemini) WithToolChoice(toolName string) *Gemini {
	g.generateConfig.ToolConfig = &genai.ToolConfig{&genai.FunctionCallingConfig{
		Mode:                 genai.FunctionCallingConfigModeAny,
		AllowedFunctionNames: []string{toolName},
	}}
	return g
}

func (g *Gemini) WithCache(cache *cache.Cache) *Gemini {
	g.cache = cache
	return g
}

type GenerateOpts struct {
	Project  string
	Location string
	Model    Model
	Cred     *google.Credentials
	Config   *genai.GenerateContentConfig
}

func New(ctx context.Context, opts GenerateOpts) *Gemini {
	gemini := &Gemini{}
	gemini.ctx = ctx
	gemini.model = opts.Model
	gemini.functions = make(map[string]Function)

	gemini.client, _ = genai.NewClient(ctx, &genai.ClientConfig{
		Project:     opts.Project,
		Location:    opts.Location,
		Backend:     genai.BackendVertexAI,
		Credentials: opts.Cred,
	})
	if opts.Config == nil {
		opts.Config = &genai.GenerateContentConfig{}
	}

	gemini.generateConfig = opts.Config
	if opts.Config.Temperature == nil {
		opts.Config.Temperature = genai.Ptr(0.5)
	}
	if opts.Config.TopK == nil {
		opts.Config.TopK = genai.Ptr(1.0)
	}
	if opts.Config.TopP == nil {
		opts.Config.TopP = genai.Ptr(0.5)
	}

	return gemini
}

func (g *Gemini) GetGenerateContentConfig() *genai.GenerateContentConfig {
	return g.generateConfig
}

func (g *Gemini) WithSafety(s []*genai.SafetySetting) *Gemini {
	g.generateConfig.SafetySettings = s
	return g
}

func (g *Gemini) GetTools() []*genai.Tool {
	return g.generateConfig.Tools
}

func (g *Gemini) GetTokenCount() int64 {
	return g.TotalTokens
}

func (g *Gemini) getCache(ctx context.Context, t *thread.Thread) (*cache.Result, error) {
	messages := t.UserQuery()
	cacheQuery := strings.Join(messages, "\n")
	cacheResult, err := g.cache.Get(ctx, cacheQuery)
	if err != nil {
		return cacheResult, err
	}

	t.AddMessage(thread.NewAssistantMessage().AddContent(
		thread.NewTextContent(strings.Join(cacheResult.Answer, "\n")),
	))

	return cacheResult, nil
}

func (g *Gemini) setCache(ctx context.Context, t *thread.Thread, cacheResult *cache.Result) error {
	lastMessage := t.LastMessage()

	if lastMessage.Role != thread.RoleAssistant || len(lastMessage.Contents) == 0 {
		return nil
	}

	contents := make([]string, 0)
	for _, content := range lastMessage.Contents {
		if content.Type == thread.ContentTypeText {
			contents = append(contents, content.Data.(string))
		} else {
			contents = make([]string, 0)
			break
		}
	}

	err := g.cache.Set(ctx, cacheResult.Embedding, strings.Join(contents, "\n"))
	if err != nil {
		return err
	}

	return nil
}

func (g *Gemini) Generate(ctx context.Context, t *thread.Thread) error {
	if t == nil {
		return nil
	}
	var err error
	var cacheResult *cache.Result
	if g.cache != nil {
		cacheResult, err = g.getCache(ctx, t)
		if err == nil {
			return nil
		} else if !errors.Is(err, cache.ErrCacheMiss) {
			return fmt.Errorf("%w: %w", ErrGeminiChat, err)
		}
	}
	var (
		errGen       error
		partContents []*genai.Content
	)
	defer func() {
		g.TotalTokens = 0
	}()

	if g.client != nil {
		partContents = g.buildRequest(t)
		if g.streamCallbackFn != nil {
			errGen = g.stream(ctx, t, partContents)
		} else {
			errGen = g.generate(ctx, t, partContents)
		}
		if errGen != nil {
			return errGen
		}
	}

	if g.cache != nil {
		err = g.setCache(ctx, t, cacheResult)
		if err != nil {
			return fmt.Errorf("%w: %w", ErrGeminiChat, err)
		}
	}

	return nil
}

func (g *Gemini) stream(ctx context.Context, t *thread.Thread, parts []*genai.Content) error {
	if len(parts) > 1 {
		systemPrompt := parts[:1]
		parts = parts[1:]
		g.generateConfig.SystemInstruction = systemPrompt[0]
	}

	//iter := g.genModel.GenerateContentStream(ctx, parts...)
	iterItems := g.client.Models.GenerateContentStream(ctx, g.model.String(), parts, g.GetGenerateContentConfig())

	var (
		messages            []*thread.Message
		currentFuncToolCall genai.FunctionCall
		allFuncToolCall     []genai.FunctionCall
		content             strings.Builder
	)

	for response, err := range iterItems {

		if response == nil || err != nil {
			return fmt.Errorf("%w", err)
		}

		if len(response.Candidates) == 0 {
			out, _ := json.Marshal(response.PromptFeedback)
			return fmt.Errorf("no candidates retured | prompt feedback: %s", string(out))
		}

		if response.UsageMetadata != nil {
			g.TotalTokens += response.UsageMetadata.TotalTokenCount
		}

		//check func tool call
		part := response.Candidates[0].Content.Parts[0]
		if part.FunctionCall != nil {
			funCall := part.FunctionCall
			allFuncToolCall = append(allFuncToolCall, *funCall)
			currentFuncToolCall = *funCall
		} else {
			content.WriteString(PartsTostring(response.Candidates[0].Content.Parts))
			g.streamCallbackFn(PartsTostring(response.Candidates[0].Content.Parts))
		}
	}

	//when iterator ends
	g.streamCallbackFn(EOS)
	if content.Len() > 0 {
		messages = append(messages, thread.NewAssistantMessage().AddContent(
			thread.NewTextContent(strings.TrimSpace(content.String())),
		))
	}

	if currentFuncToolCall.Name != "" {
		messages = append(messages, functionToolCallsToToolCallMessage(allFuncToolCall))
		messages = append(messages, g.callFuncTools(allFuncToolCall)...)
	}

	t.AddMessages(messages...)
	return nil
}

func (g *Gemini) generate(ctx context.Context, t *thread.Thread, parts []*genai.Content) error {

	response, err := g.client.Models.GenerateContent(ctx, g.model.String(), parts, g.GetGenerateContentConfig())
	if err != nil {
		return fmt.Errorf("%w: %w", ErrGeminiChat, err)
	}

	if len(response.Candidates) == 0 {
		out, _ := json.Marshal(response.PromptFeedback)
		return fmt.Errorf("no candidates retured | prompt feedback: %s", string(out))
	}
	if response.UsageMetadata != nil {
		g.TotalTokens += response.UsageMetadata.TotalTokenCount
	}

	var messages []*thread.Message

	//check func tool call
	part := response.Candidates[0].Content.Parts[0]
	if part.FunctionCall != nil {
		funCall := *part.FunctionCall
		messages = append(messages, functionToolCallsToToolCallMessage([]genai.FunctionCall{funCall}))
		messages = append(messages, g.callFuncTools([]genai.FunctionCall{funCall})...)
	} else {
		messages = []*thread.Message{
			thread.NewAssistantMessage().AddContent(
				thread.NewTextContent(PartsTostring(response.Candidates[0].Content.Parts)),
			),
		}
	}
	t.Messages = append(t.Messages, messages...)
	return nil
}

func (g *Gemini) buildRequest(t *thread.Thread) []*genai.Content {
	return g.threadToPartContentMessage(t)
}

func (g *Gemini) callFuncTools(toolCalls []genai.FunctionCall) []*thread.Message {
	if len(g.functions) == 0 || len(toolCalls) == 0 {
		return nil
	}

	var messages []*thread.Message
	for _, toolCall := range toolCalls {
		result, err := g.callTool(toolCall)
		if err != nil {
			result = fmt.Sprintf("error: %s", err)
		}

		messages = append(messages, toolCallResultToThreadMessage(toolCall, result))
	}

	return messages
}

func (g *Gemini) callTool(fnc genai.FunctionCall) (string, error) {
	fn, ok := g.functions[fnc.Name]
	if !ok {
		return "", fmt.Errorf("unknown function %s", fnc.Name)
	}

	jsonArgs, err := json.Marshal(fnc.Args)
	if err != nil {
		return "", fmt.Errorf("error in marshal: %w", err)
	}

	resultAsJSON, err := callFnWithArgumentAsJSON(fn.Fn, string(jsonArgs))
	if err != nil {
		return "", err
	}

	return resultAsJSON, nil
}
