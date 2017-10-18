package annoy

import (
	"syscall"
)

type Locker interface {
	ReadLock(uintptr, int64, int64) error
	WriteLock(uintptr, int64, int64) error
	UnLock(uintptr, int64, int64) error
}

func newLocker() Locker {
	return Flock{}
}

type Flock struct {
}

func (f Flock) ReadLock(fd uintptr, start, len int64) error {
	return f.flock(fd, syscall.LOCK_SH)
}

func (f Flock) WriteLock(fd uintptr, start, len int64) error {
	return f.flock(fd, syscall.LOCK_EX)
}

func (f Flock) UnLock(fd uintptr, start, len int64) error {
	return f.flock(fd, syscall.LOCK_UN)
}

func (f Flock) flock(fd uintptr, how int) error {
	return syscall.Flock(int(fd), how)
}
