package gemini

import (
	"encoding/json"
	"fmt"
	"github.com/henomis/lingoose/thread"
	"google.golang.org/genai"
	"strings"
)

//func threadToPartContentMessage(t *thread.Thread) []*genai.Content {
//	var chatMessages []*genai.Content
//
//	//msgToModel = system prompts + user utterance
//	for _, m := range t.Messages {
//		switch m.Role {
//		case thread.RoleUser, thread.RoleSystem, thread.RoleAssistant:
//			for _, content := range m.Contents {
//				contentData, ok := content.Data.(string)
//				if !ok {
//					continue
//				}
//				chatMessages = append(chatMessages, genai.Text(contentData)...)
//			}
//		//case thread.RoleAssistant:
//		//	continue
//		case thread.RoleTool:
//			if data, isTollResponseData := m.Contents[0].Data.(thread.ToolResponseData); isTollResponseData && !m.Contents[0].Processed {
//				var funcResponses genai.FunctionResponse
//				funcResponses.Name = data.Name
//				funcResponses.Response = map[string]any{
//					"result": data.Result,
//				}
//				chatMessages = append(chatMessages, funcResponses)
//			}
//		}
//	}
//	return chatMessages
//}

func (g *Gemini) threadToPartContentMessage(t *thread.Thread) []*genai.Content {
	var (
		contentMessages []*genai.Content
	)

	for _, m := range t.Messages[:len(t.Messages)-1] {
		switch m.Role {
		case thread.RoleSystem:
			g.generateConfig.SystemInstruction = &genai.Content{
				Role:  "system_instructions",
				Parts: []*genai.Part{{Text: m.Contents[0].AsString()}}}
			//fmt.Println("----System-----")
			//fmt.Println(m.Contents[0].AsString()[:100])
			//fmt.Println("----End----")

		case thread.RoleUser:
			role := threadRoleToGeminiRole[thread.RoleUser]
			contentMessages = append(contentMessages, formChatHistory(role, m)...)

		case thread.RoleAssistant:
			assistantRole := threadRoleToGeminiRole[thread.RoleAssistant]
			contentMessages = append(contentMessages, formChatHistory(assistantRole, m)...)

		case thread.RoleTool:
			toolRole := threadRoleToGeminiRole[thread.RoleTool]
			contentMessages = append(contentMessages, formChatHistory(toolRole, m)...)
		}
	}

	for _, content := range t.LastMessage().Contents {
		switch v := content.Data.(type) {
		case thread.ToolResponseData:
			var response map[string]any
			_ = json.Unmarshal([]byte(v.Result), &response)
			contentMessages = append(contentMessages, &genai.Content{
				Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
					Name:     v.Name,
					Response: response}},
				}})
		default:
			contentMessages = append(contentMessages, genai.Text(content.AsString())...)
		}
	}

	/*	fmt.Println("----History----")
		for _, history := range contentMessages {
			fmt.Println(history.Role, history.Parts)
		}
		fmt.Println("----Messages----")
		fmt.Println(chatMessages)
		fmt.Println("----End----")
		g.session = g.genModel.StartChat()
		g.session.History = contentMessages
	*/

	return contentMessages
}

func PartsTostring(parts []*genai.Part) string {
	var msg strings.Builder
	size := len(parts) - 1
	for i := 0; i < len(parts); i++ {
		if parts[i].Text != "" {
			msg.WriteString(fmt.Sprintf("%s", parts[i].Text))
			if i != size {
				msg.WriteString(" ")
			}
		}
		if parts[i].FunctionCall != nil {
			fp := parts[i].FunctionCall
			msg.WriteString(fmt.Sprintf("FunctionCall: %+v ", fp))
		}

		if parts[i].FunctionResponse != nil {
			fp := parts[i].FunctionResponse
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
		switch v := content.Data.(type) {
		case []thread.ToolCallData:
			for _, tcd := range v {
				var args map[string]any
				_ = json.Unmarshal([]byte(tcd.Arguments), &args)
				chatContent.Parts = append(chatContent.Parts, &genai.Part{FunctionCall: &genai.FunctionCall{
					Name: tcd.Name,
					Args: args,
				}})
			}
		case thread.ToolResponseData:
			var toolResponse map[string]any
			_ = json.Unmarshal([]byte(v.Result), &toolResponse)
			chatContent.Parts = append(chatContent.Parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{
				Name:     v.Name,
				Response: toolResponse,
			}})
		default:
			chatContent.Parts = append(chatContent.Parts, &genai.Part{Text: content.AsString()})
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
