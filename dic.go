package gortex

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

const UNK = "UnK"
const BOS = "BoS"
const EOS = "EoS"

type Token struct {
	Token     string
	Frequency uint
}

type Dictionary struct {
	Token2ID        map[string]uint
	Token2Frequency map[string]uint
	id2Token        []string `json: omit`
}

func NewDictionary() *Dictionary {
	dic := &Dictionary{Token2ID: make(map[string]uint), Token2Frequency: make(map[string]uint)}
	dic.Add(UNK)
	dic.Token2Frequency[UNK] = 10e9
	dic.Add(BOS)
	dic.Token2Frequency[BOS] = 10e9
	dic.Add(EOS)
	dic.Token2Frequency[EOS] = 10e9
	return dic
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
		dic.Token2Frequency[t.Token] = d.Token2Frequency[t.Token]
	}

	return dic
}

func (d *Dictionary) Add(token string) {
	d.Token2Frequency[token]++
	_, ok := d.Token2ID[token]
	if !ok {
		d.Token2ID[token] = uint(len(d.Token2ID))
	}
}

func (d *Dictionary) Save(name string) error {
	f, err := os.Create(name)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(f)
	err = encoder.Encode(d)
	if err != nil {
		return err
	}
	f.Close()
	return nil
}

func (d *Dictionary) Load(name string) error {
	if f, e := os.Open(name); e != nil {
		return e
	} else {
		defer f.Close()
		decoder := json.NewDecoder(f)
		if e = decoder.Decode(d); e != nil {
			return e
		}
	}
	return nil
}

func (d *Dictionary) Print(n int) {
	n = MinInt(n, d.Len())
	for i := range make([]struct{}, n) {
		token := d.TokenByID(uint(i))
		fmt.Printf("%d %s %d\n", i, token, d.Token2Frequency[token])
	}
}

func SaveDictionary(name string, dic *Dictionary) error {
	return dic.Save(name)
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

func (d *Dictionary) Encode(tokenizer Tokenizer, text string) []uint {
	tokens := tokenizer.Split(text)
	encoded := make([]uint, len(tokens))
	for i := range tokens {
		encoded[i] = d.IDByToken(tokens[i])
	}
	return encoded
}
func (d *Dictionary) Decode(sequence []uint, delimiter string) string {
	// read sample
	source := ""
	for i := range sequence {
		if i > 0 {
			source += delimiter
		}
		source += d.TokenByID(sequence[i])
	}
	return source
}

func (d Dictionary) FromFile(file string, s Tokenizer) error {
	if f, e := os.Open(file); e != nil {
		return e
	} else {
		defer f.Close()
		r := bufio.NewReader(f)
		for {
			if line, e := r.ReadString('\n'); e != nil {
				break
			} else {
				line = strings.TrimSpace(line) // remove ending \n
				if len(line) > 0 {
					for _, token := range s.Split(line) {
						d.Add(token)
					}
				}
			}
		}
	}
	return nil
}

func DictionaryFromFile(file string, s Tokenizer) (*Dictionary, error) {
	f, e := os.Open(file)
	if e != nil {
		return nil, e
	}
	r := bufio.NewReader(f)
	dic := NewDictionary()
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			break
		}
		line = strings.TrimSpace(line) // remove ending \n
		if len(line) > 0 {
			for _, token := range s.Split(line) {
				dic.Add(token)
			}
		}
	}

	f.Close()
	return dic, nil
}
