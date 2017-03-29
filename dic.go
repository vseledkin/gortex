package gortex

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

const UNK = "<unk>"

type Dictionary struct {
	Token2ID map[string]int
	id2Token []string `json: omit`
}

func (d *Dictionary) String() string {
	str := ""
	for k, v := range d.Token2ID {
		str += fmt.Sprintf("%s:%d\n", k, v)
	}
	return str
}

func (d *Dictionary) TokenByID(id int) string {
	if len(d.id2Token) == 0 {
		d.id2Token = make([]string, len(d.Token2ID))
		for k, v := range d.Token2ID {
			d.id2Token[v] = k
		}
	}
	return d.id2Token[id]
}

func (d *Dictionary) IDByToken(token string) int {
	if id, ok := d.Token2ID[token]; ok {
		return id
	}
	return d.Token2ID[UNK]
}

//Len number of tokens in dictionary
func (d *Dictionary) Len() int {
	return len(d.Token2ID)
}

func DictionaryFromFile(file string, s Tokenizer) (*Dictionary, error) {
	f, e := os.Open(file)
	if e != nil {
		return nil, e
	}
	r := bufio.NewReader(f)
	dic := &Dictionary{Token2ID: make(map[string]int)}
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			break
		}
		line = strings.TrimSpace(line)
		if len(line) > 0 {
			for _, token := range s.Split(line) {
				_, ok := dic.Token2ID[token]
				if !ok {
					dic.Token2ID[token] = len(dic.Token2ID)
				}
			}
		}
	}
	f.Close()
	//if _, ok := dic.Token2ID[UNK]; !ok {
	//	dic.Token2ID[UNK] = len(dic.Token2ID)
	//}
	return dic, nil
}
