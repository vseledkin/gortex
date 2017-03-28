package gortex

import (
	"bufio"
	"os"
	"strings"
)

func SampleVisitor(file string, tokenizer Tokenizer, dictionary *Dictionary, visitor func(sample []int)) error {
	f, e := os.Open(file)
	if e != nil {
		return e
	}
	r := bufio.NewReader(f)
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			break
		}
		line = strings.TrimSpace(line)
		if len(line) > 0 {
			terms := tokenizer.Split(line)
			sample := make([]int, len(terms))
			for i, term := range terms {
				sample[i] = dictionary.IDByToken(term)
			}
			visitor(sample)
		}
	}
	f.Close()
	return nil
}
