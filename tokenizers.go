package gortex

import "strings"

//Tokenizer interface to string tokenizer
type Tokenizer interface {
	Split(str string) []string
}

//WhiteSpaceSplitter space delimited tokens
type WhiteSpaceSplitter struct{}

func (s WhiteSpaceSplitter) Split(str string) []string {
	return strings.Fields(str)
}

//CharSplitter space delimited tokens
type CharSplitter struct{}

func (s CharSplitter) Split(str string) []string {
	runes := []rune(str)
	ret := make([]string, len(runes))
	for i, r := range runes {
		ret[i] = string(r)
	}
	return ret
}
