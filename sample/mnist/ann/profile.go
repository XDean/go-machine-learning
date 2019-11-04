package mnist_ann

type Profile struct {
	Total int
	Hit   int

	Index     int
	Recent    []bool
	RecentHit int
}

func NewProfile(recent int) *Profile {
	return &Profile{Recent: make([]bool, recent)}
}

func (p *Profile) Add(b bool) {
	p.Total++
	if b {
		p.Hit++
		p.RecentHit++
	}
	if p.Recent[p.Index] {
		p.RecentHit--
	}
	p.Recent[p.Index] = b
	p.Index = (p.Index + 1) % len(p.Recent)
}

func (p *Profile) HitRate() float64 {
	return float64(p.Hit) / float64(p.Total)
}

func (p *Profile) RecentHitRate() float64 {
	if p.Total < len(p.Recent) {
		return float64(p.RecentHit) / float64(p.Total)
	}
	return float64(p.RecentHit) / float64(len(p.Recent))
}
