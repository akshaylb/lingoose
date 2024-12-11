package gemini

import (
	"encoding/json"
	"fmt"
	"strings"

	"cloud.google.com/go/vertexai/genai"
	"github.com/henomis/lingoose/thread"
)

func threadToPartMessage(t *thread.Thread) []genai.Part {
	var chatMessages []genai.Part

	//msgToModel = system prompts + user utterance
	for _, m := range t.Messages {
		switch m.Role {
		case thread.RoleUser, thread.RoleSystem, thread.RoleAssistant:
			for _, content := range m.Contents {
				contentData, ok := content.Data.(string)
				if !ok {
					continue
				}
				chatMessages = append(chatMessages, genai.Text(contentData))
			}
		//case thread.RoleAssistant:
		//	continue
		case thread.RoleTool:
			if data, isTollResponseData := m.Contents[0].Data.(thread.ToolResponseData); isTollResponseData && !m.Contents[0].Processed {
				var funcResponses genai.FunctionResponse
				funcResponses.Name = data.Name
				funcResponses.Response = map[string]any{
					"result": data.Result,
				}
				chatMessages = append(chatMessages, funcResponses)
			}
		}
	}
	return chatMessages
}

func (g *Gemini) threadToChatPartMessage(t *thread.Thread) ([]genai.Part, error) {
	var (
		chatMessages []genai.Part
		chatHistory  []*genai.Content
	)

	for _, m := range t.Messages {
		if m == t.LastMessage() && (m.Role == thread.RoleUser) {
			break
		}

		switch m.Role {
		case thread.RoleSystem:
			g.genModel.SystemInstruction = &genai.Content{
				Role:  "system_instructions",
				Parts: []genai.Part{genai.Text(m.Contents[0].AsString())},
			}

		case thread.RoleUser:
			role := threadRoleToGeminiRole[thread.RoleUser]
			chatHistory = append(chatHistory, formChatHistory(role, m)...)

		case thread.RoleAssistant:
			assistantRole := threadRoleToGeminiRole[thread.RoleAssistant]
			chatHistory = append(chatHistory, formChatHistory(assistantRole, m)...)

		case thread.RoleTool:
			continue
		}
	}

	for _, content := range t.LastMessage().Contents {
		switch v := content.Data.(type) {
		case thread.ToolResponseData:
			var response map[string]any
			_ = json.Unmarshal([]byte(v.Result), &response)
			chatMessages = append(chatMessages, genai.FunctionResponse{
				Name:     v.Name,
				Response: response,
			})
		default:
			chatMessages = append(chatMessages, genai.Text(content.AsString()))
		}
	}

	g.session = g.genModel.StartChat()
	g.session.History = chatHistory
	return chatMessages, nil
}

func PartsTostring(parts []genai.Part) string {
	var msg strings.Builder
	size := len(parts) - 1
	for i := 0; i < len(parts); i++ {
		switch parts[i].(type) {
		case genai.Text:
			msg.WriteString(fmt.Sprintf("%v", parts[i]))
			if i != size {
				msg.WriteString(" ")
			}
		case genai.FunctionCall:
			fp := parts[i].(genai.FunctionCall)
			msg.WriteString(fmt.Sprintf("FunctionCall: %+v ", fp))

		case genai.FunctionResponse:
			fp := parts[i].(genai.FunctionResponse)
			msg.WriteString(fmt.Sprintf("FunctionResponse: %+v ", fp))
		}
	}
	return msg.String()
}

func functionToolCallsToToolCallMessage(toolCalls []genai.FunctionCall) *thread.Message {
	if len(toolCalls) == 0 {
		return nil
	}

	var toolCallData []thread.ToolCallData
	for _, toolCall := range toolCalls {
		args, _ := json.Marshal(toolCall.Args)
		toolCallData = append(toolCallData, thread.ToolCallData{
			Name:      toolCall.Name,
			Arguments: string(args),
		})
	}

	return thread.NewAssistantMessage().AddContent(
		thread.NewToolCallContent(
			toolCallData,
		),
	)
}

func toolCallResultToThreadMessage(fnCall genai.FunctionCall, result string) *thread.Message {
	return thread.NewToolMessage().AddContent(
		thread.NewToolResponseContent(
			thread.ToolResponseData{
				Name:   fnCall.Name,
				Result: result,
			},
		),
	)
}

func formChatHistory(role string, m *thread.Message) (ch []*genai.Content) {
	chatContent := &genai.Content{
		Role: role,
	}
	for _, content := range m.Contents {
		switch content.Data.(type) {
		case []thread.ToolCallData:
			for _, tcd := range content.Data.([]thread.ToolCallData) {
				var args map[string]any
				_ = json.Unmarshal([]byte(tcd.Arguments), &args)
				chatContent.Parts = append(chatContent.Parts, genai.FunctionCall{
					Name: tcd.Name,
					Args: args,
				})
			}
		default:
			chatContent.Parts = append(chatContent.Parts, genai.Text(content.AsString()))
		}

	}
	ch = append(ch, chatContent)
	return
}

// LastUserMessage returns last user or assistant message in the thread
func LastUserMessage(t *thread.Thread) *thread.Message {
	for i := len(t.Messages) - 1; i >= 0; i-- {
		if t.Messages[i].Role == thread.RoleUser || t.Messages[i].Role == thread.RoleAssistant {
			return t.Messages[i]
		}
	}
	return nil
}
