package gemini

import "fmt"

var (
	ErrGeminiChat   = fmt.Errorf("gemini chat error")
	ErrGeminiNoChat = fmt.Errorf("gemini no chat message recieved")
)

type Model string

func (m Model) String() string {
	return string(m)
}

const (
	Gemini1Pro       Model = "gemini-1.0-pro"
	Gemini1Pro001    Model = "gemini-1.0-pro-001"
	GeminiPro15002   Model = "gemini-1.5-pro-002"
	GeminiFlash15002 Model = "gemini-1.5-flash-002"
	GeminiFlash20Exp Model = "gemini-2.0-flash-exp"
)

type StreamCallback func(string)
