package rag

import (
	"context"
	"fmt"
	"regexp"

	"encoding/json"

	"github.com/henomis/lingoose/document"
	"github.com/henomis/lingoose/index"
	"github.com/henomis/lingoose/index/option"
	"github.com/henomis/lingoose/loader"
	"github.com/henomis/lingoose/textsplitter"
	"github.com/henomis/lingoose/thread"
)

const (
	defaultChunkSize    = 1000
	defaultChunkOverlap = 0
	defaultTopK         = 5
)

type LLM interface {
	Generate(context.Context, *thread.Thread) error
}

type Loader interface {
	LoadFromSource(context.Context, string) ([]document.Document, error)
}

type RAG struct {
	index        *index.Index
	chunkSize    uint
	chunkOverlap uint
	topK         uint
	loaders      map[*regexp.Regexp]Loader // this map a regexp as string to a loader
}

type Fusion struct {
	RAG
	llm LLM
}

func New(index *index.Index) *RAG {
	rag := &RAG{
		index:        index,
		chunkSize:    defaultChunkSize,
		chunkOverlap: defaultChunkOverlap,
		topK:         defaultTopK,
		loaders:      make(map[*regexp.Regexp]Loader),
	}

	return rag.withDefaultLoaders()
}

func (r *RAG) WithChunkSize(chunkSize uint) *RAG {
	r.chunkSize = chunkSize
	return r
}

func (r *RAG) WithChunkOverlap(chunkOverlap uint) *RAG {
	r.chunkOverlap = chunkOverlap
	return r
}

func (r *RAG) WithTopK(topK uint) *RAG {
	r.topK = topK
	return r
}

func (r *RAG) withDefaultLoaders() *RAG {
	r.loaders[regexp.MustCompile(`.*\.pdf`)] = loader.NewPDFToText()
	r.loaders[regexp.MustCompile(`.*\.docx`)] = loader.NewLibreOffice()
	r.loaders[regexp.MustCompile(`.*\.txt`)] = loader.NewText()

	return r
}

func (r *RAG) WithLoader(sourceRegexp *regexp.Regexp, loader Loader) *RAG {
	r.loaders[sourceRegexp] = loader
	return r
}

func (r *RAG) AddSources(ctx context.Context, sources ...string) error {
	for _, source := range sources {
		documents, err := r.addSource(ctx, source)
		if err != nil {
			return err
		}

		err = r.index.LoadFromDocuments(ctx, documents)
		if err != nil {
			return err
		}
	}

	return nil
}

func (r *RAG) chunkDocuments(documents ...document.Document) []document.Document {
	var newDocs []document.Document
	//chunkSize := 2000

	for _, doc := range documents {
		var parsed map[string]interface{}

		metadata := doc.Metadata
		content := doc.Content

		err := json.Unmarshal([]byte(content), &parsed)
		if err != nil {
			return nil
		}

		for k, part := range parsed {
			partString, err := json.Marshal(part)
			if err != nil {
				return nil
			}
			var newDoc = document.Document{}
			m, err := json.Marshal(map[string]string{k: string(partString)})

			newDoc.Content = string(m)
			newDoc.Metadata = metadata
			newDocs = append(newDocs, newDoc)
		}

	}
	print(len(newDocs))
	return newDocs
}

func (r *RAG) AddDocuments(ctx context.Context, documents ...document.Document) error {
	chunkedDocuments := r.chunkDocuments(documents...)
	return r.index.LoadFromDocuments(ctx, chunkedDocuments)
}

func (r *RAG) Retrieve(ctx context.Context, query string) ([]string, error) {
	results, err := r.index.Query(ctx, query, option.WithTopK(int(r.topK)))
	var resultsAsString []string
	for _, result := range results {
		resultsAsString = append(resultsAsString, result.Content())
	}

	return resultsAsString, err
}

func (r *RAG) addSource(ctx context.Context, source string) ([]document.Document, error) {
	var sourceLoader Loader
	for regexpStr, loader := range r.loaders {
		if regexpStr.MatchString(source) {
			sourceLoader = loader
		}
	}

	if sourceLoader == nil {
		return nil, fmt.Errorf("unsupported source type")
	}

	documents, err := sourceLoader.LoadFromSource(ctx, source)
	if err != nil {
		return nil, err
	}

	return textsplitter.NewRecursiveCharacterTextSplitter(
		int(r.chunkSize),
		int(r.chunkOverlap),
	).SplitDocuments(documents), nil
}
