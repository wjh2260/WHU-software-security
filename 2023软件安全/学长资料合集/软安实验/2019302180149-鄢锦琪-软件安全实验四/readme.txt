exp4.exe为老师给定的栈溢出程序
在测试时选择导入final.txt即可.

中间文件和中间程序中是实验报告中提到的所有文件
其中:
	rop_chains.txt        mona生成的rop链的文件
                arguments.txt         表示数据区内容
	shellcode.asm         shellcode的汇编代码
	inline_asm.cpp        编译后用ida获取字节码,a函数用于生成arguments.txt
	shellcode.txt         shellcode的字符串格式的字节码
	process_hex.py        处理部分字节码的脚本
	shellcode(0xhex).txt  shellcode的0x格式的字节码
	create_final.py       用于生成final.txt文件, 整合所有字节码,生成payload文件
	