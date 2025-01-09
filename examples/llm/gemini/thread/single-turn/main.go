package main

import (
	"context"
	"fmt"
	"github.com/henomis/lingoose/llm/gemini"
	"github.com/henomis/lingoose/thread"
	"google.golang.org/genai"
	"os"
)

var (
	PROJECT      = "conversenow-dev"
	REGION       = "us-central1"
	GCP_KEY_PATH string
)

func init() {
	GCP_KEY_PATH = os.Getenv("GCP_KEY_PATH")
}

func main() {
	ctx := context.Background()
	var err error
	//client, err := genai.NewClient(ctx, PROJECT, REGION, option.WithCredentialsFile(GCP_KEY_PATH))
	//if err != nil {
	//	return
	//}

	//CredJson, err := ioutil.ReadFile(GCP_KEY_PATH)
	//if err != nil {
	//	fmt.Println(err)
	//	return
	//}
	//cred, err := google.CredentialsFromJSON(ctx, CredJson)
	//if err != nil {
	//	fmt.Println(err)
	//	return
	//}

	geminiLLM := gemini.New(ctx, gemini.GenerateOpts{
		Project:  PROJECT,
		Location: REGION,
		Model:    gemini.GeminiFlash20Exp,
		Cred:     nil,
		Config:   &genai.GenerateContentConfig{},
	})

	t := thread.New().AddMessage(
		thread.NewUserMessage().AddContent(
			thread.NewTextContent("Hello, I'm a user"),
		).AddContent(
			thread.NewTextContent("Can you greet me?"),
		),
	).AddMessage(
		thread.NewUserMessage().AddContent(
			thread.NewTextContent("please greet me as a pirate."),
		),
	)
	fmt.Println("INPUT THREAD ::")
	fmt.Println(t.String())

	err = geminiLLM.Generate(context.Background(), t)
	if err != nil {
		panic(err)
	}

	fmt.Println("PREDICTION THREAD ::")
	fmt.Println(t.String())

	//t.ClearMessages()
	t.AddMessage(thread.NewUserMessage().AddContent(
		thread.NewTextContent("Now translate to italian given to you as a poem. Give me a single poem of your choice"),
	))

	fmt.Println("INPUT THREAD ::")
	fmt.Println(t.String())

	err = geminiLLM.Generate(context.Background(), t)
	if err != nil {
		panic(err)
	}

	fmt.Println("PREDICTION THREAD ::")
	fmt.Println(t.String())

}
