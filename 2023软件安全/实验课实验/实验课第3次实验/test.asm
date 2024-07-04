.386
.model flat,stdcall
option casemap:none;case sensetive
      include \masm32\include\windows.inc
      include \masm32\include\kernel32.inc
      include \masm32\include\user32.inc
      includelib \masm32\lib\kernel32.lib
      includelib \masm32\lib\user32.lib

.data
szCap   db "教学测试",0
szMsgOK1 db "PE入口点测试1:进入第一入口位置401000H!",0
szMsgOK2 db "PE入口点测试2:进入第一入口位置401016H!",0

.code
start:
    push eax
    push ebx
    push ecx
    invoke MessageBox,NULL,addr szMsgOK1,addr szCap,40h+1000h
    push eax
    invoke MessageBox,NULL,addr szMsgOK2,addr szCap,40h+1000h
    xor eax,eax
    xor ecx,eax
    invoke ExitProcess,NULL
end start