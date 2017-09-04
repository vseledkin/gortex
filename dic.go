package gortex

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

const UNK = "<unk>"

type Token struct {
	Token     string
	Frequency uint
}

type Dictionary struct {
	Token2ID        map[string]uint
	Token2Frequency map[string]uint
	id2Token        []string `json: omit`
}

func (d *Dictionary) AllWordsKnown(text string, s Tokenizer) bool {
	for _, token := range s.Split(text) {
		_, ok := d.Token2ID[token]
		if !ok {
			return false
		}
	}
	return true
}

func (d *Dictionary) Top(n uint) *Dictionary {
	if uint(d.Len()) <= n {
		return d
	}
	dic := &Dictionary{Token2ID: make(map[string]uint), Token2Frequency: make(map[string]uint)}
	var sorted []*Token
	for k, v := range d.Token2Frequency {
		sorted = append(sorted, &Token{k, v})
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Frequency > sorted[j].Frequency
	})

	for id, t := range sorted[:n] {
		dic.Token2ID[t.Token] = uint(id)
	}

	return dic
}

func SaveDictionary(name string, dic *Dictionary) error {
	// save MODEL_NAME
	f, err := os.Create(name)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(f)
	err = encoder.Encode(dic)
	if err != nil {
		return err
	}
	f.Close()
	return nil
}

func LoadDictionary(name string) (*Dictionary, error) {

	if len(name) == 0 {
		return nil, fmt.Errorf("No dictionary file provided! [%s]", name)
	}
	f, e := os.Open(name)
	if e != nil {
		return nil, e
	}
	var m *Dictionary
	decoder := json.NewDecoder(f)
	e = decoder.Decode(&m)
	if e != nil {
		return nil, e
	}
	f.Close()
	return m, nil
}
func (d *Dictionary) String() string {
	str := ""
	for k, v := range d.Token2ID {
		str += fmt.Sprintf("%s:%d\n", k, v)
	}
	return str
}

func (d *Dictionary) TokenByID(id uint) string {
	if len(d.id2Token) == 0 {
		d.id2Token = make([]string, len(d.Token2ID))
		for k, v := range d.Token2ID {
			d.id2Token[v] = k
		}
	}
	return d.id2Token[id]
}

func (d *Dictionary) IDByToken(token string) uint {
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
	dic := &Dictionary{Token2ID: make(map[string]uint), Token2Frequency: make(map[string]uint)}
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			break
		}
		//line = strings.TrimSpace(line) // remove ending \n
		if len(line) > 0 {
			for _, token := range s.Split(line) {
				dic.Token2Frequency[token]++
				_, ok := dic.Token2ID[token]
				if !ok {
					dic.Token2ID[token] = uint(len(dic.Token2ID))
				}
			}
		}
	}
	f.Close()
	return dic, nil
}
