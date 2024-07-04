#include <stdio.h>
#include <string.h>
#include <windows.h>

int main(int argc, char **argv) {
    char* findPath = NULL;

    // 检查命令行参数个数
    if (argc >= 2) {
        // 处理只有一个参数的情况
        if (argc == 2) {
            // 获取目标路径
            findPath = argv[1];
            // 拼接目标路径的通配符
            char* targetfile = "\\*.exe";
            strcat(findPath, targetfile);

            // 查找第一个符合条件的文件
            WIN32_FIND_DATAA p; // 使用ANSI版本的WIN32_FIND_DATAA
            HANDLE h = FindFirstFileA(findPath, &p);

            // 存储搜索到的文件名
            wchar_t result[1000]; // 使用宽字符数组
            wcscpy(result, L""); // 初始化为空字符串
            // 存储临时文件名，用于连接到result后面
            wchar_t temp[100];
            mbstowcs(temp, p.cFileName, sizeof(temp) / sizeof(temp[0])); // 转换为宽字符
            wcscat(result, temp);
            wcscat(result, L"\n");

            // 遍历目录下的所有符合条件的文件
            while (FindNextFileA(h, &p)) {
                // 递归查找目录的子目录
                mbstowcs(temp, p.cFileName, sizeof(temp) / sizeof(temp[0])); // 转换为宽字符
                wcscat(result, temp);
                wcscat(result, L"\n");
            }

            // 弹出消息框显示搜索结果
            MessageBoxW(NULL, result, L"搜索到exe文件如下", MB_OKCANCEL);
        } else {
            // 参数错误，弹出错误消息框
            MessageBoxA(NULL, "参数错误", "Error", MB_ICONERROR); // 使用ANSI版本的MessageBoxA
        }
    } else {
        // 未指明路径，弹出错误消息框
        MessageBoxA(NULL, "未指明路径", "Error", MB_ICONERROR); // 使用ANSI版本的MessageBoxA
    }

    return 0;
}
