% Notice: this template can be only used in Windows System.
% Otherwise please change the absolute path of line 38 to the Simkai.ttf file location.
%
% Author: LiPtP
% Compiler (no references): XeLaTeX
% Compiler (with references): XeLaTeX -> BibTeX -> XeLaTeX -> XeLaTeX

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{SEU-AI-Report}[Report Template for AIDL Course V1.0]

% 基础类
\LoadClass[a4paper,12pt]{article}


\RequirePackage[backend=bibtex,style=gb7714-2015,sorting=none]{biblatex}
\RequirePackage{xeCJK}
\RequirePackage{fontspec}
\RequirePackage{titlesec}
\RequirePackage{zhnumber}
\RequirePackage{subfig}
\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{geometry}        % 页面设置
\RequirePackage{xcolor}          % For defining custom colors
\RequirePackage{listings}        % For code highlighting
\RequirePackage{lipsum}          % For Dummy Content
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
\RequirePackage{setspace}
\RequirePackage{booktabs}
\RequirePackage{float}
\RequirePackage{xcolor}
\RequirePackage{hyperref}

% 超链接设定
\hypersetup{
    colorlinks=true,
    linkcolor=red, % 超链接颜色设为红色
    citecolor=blue, % 参考文献超链接颜色设为蓝色
    urlcolor=cyan
}


% 中英文字体
\setmainfont{Times New Roman}
\setCJKmainfont{SimSun}
\setCJKsansfont{SimHei}
\setCJKfamilyfont{hei}{SimHei}
\setCJKfamilyfont{kai}[Path=C:/Windows/Fonts/]{SimKai.ttf}

% 页边距、正文行距
\geometry{left=2.5cm,right=2.5cm,top=3.5cm,bottom=2.5cm}
\renewcommand{\baselinestretch}{1.2}  % 设置行距为1.2倍
% 标题
\renewcommand{\title}[2]{ % 仅接收实验编号
    \begin{center}
        {\large \CJKfamily{hei}《人工智能与深度学习》课程 \\[2em] \LARGE 实~验~报~告} \\[2em]
        {\large \CJKfamily{kai} （第~#1~次实验）} \\[4em]
    \end{center}
    % \vfill
    \studentinfo
    \begin{center}
        \large\CJKfamily{hei} 实验#1： #2 \hfill 
    \end{center}

   \noindent\rule{\textwidth}{0.4pt} 
   \vspace{0.5em} 
}

\newcommand{\studentinfo}{ % 直接定义信息，无需额外参数
  \begin{center}
    {\large 学号：\studentid} \\[2em]
    {\large 姓名：\studentname} \\[2em]
    {\large 日期：\studentyear~年~\studentmonth~月~\studentday~日} \\[2em]
  \end{center}
%   \vspace{2em}
}

% 设置学生信息
\newcommand{\setstudentinfo}[5]{ % 先调用这个命令来设置信息
    \renewcommand{\studentid}{#1}
    \renewcommand{\studentname}{#2}
    \renewcommand{\studentyear}{#3}
    \renewcommand{\studentmonth}{#4}
    \renewcommand{\studentday}{#5}
}

% 学生信息缺省值
\newcommand{\studentid}{未填写}
\newcommand{\studentname}{未填写}
\newcommand{\studentyear}{未填写}
\newcommand{\studentmonth}{未填写}
\newcommand{\studentday}{未填写}

% 主文档字体、章节编号设置
\titleformat{\section}{\large\bfseries\CJKfamily{hei}}{\thesection、}{0em}{}
\titleformat{\subsection}{\normalsize\bfseries\CJKfamily{hei}}{\thesubsection}{1em}{}
\renewcommand{\normalsize}{\fontsize{12pt}{14pt}\selectfont}
\renewcommand{\thesection}{\zhnum{section}}
\renewcommand{\thesubsection}{\arabic{section}.\arabic{subsection}}


% 附录（默认不换页，可自行newpage）
\renewcommand\appendix{
    \newpage
    \setcounter{section}{0}
    \setcounter{subsection}{0}
    \setcounter{lstlisting}{0}
    \renewcommand\thesection{附录~\Alph{section}}
    \renewcommand\thesubsection{\Alph{section}.\arabic{subsection}}
    \titleformat{\section}{\large\bfseries\CJKfamily{hei}}{\thesection}{1em}{}
    \titleformat{\subsection}{\normalsize\bfseries\CJKfamily{hei}}{\thesubsection}{1em}{}
    }

% 参考文献支持
\addbibresource{ref.bib}
\renewcommand{\refname}{参考文献}

% 页眉页脚设置，第一页缺省页眉
\pagestyle{fancy}
\fancyhf{}
\rhead{\small \studentid~\studentname}
\lhead{\small 《人工智能与深度学习》实验报告}
\cfoot{\small \thepage~/~\pageref*{LastPage}}

\fancypagestyle{firstpage}{
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt} 
  \cfoot{\small \thepage~/~\pageref*{LastPage}}
}

\AtBeginDocument{\thispagestyle{firstpage}}

% Caption 设置
\renewcommand{\lstlistingname}{程序}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}

% Listings
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codeblue}{rgb}{0,0,0.8}
\definecolor{backcolor}{rgb}{0.95,0.95,0.95}

% Define Python code listing style
\lstdefinestyle{codestyle}{
    backgroundcolor=\color{backcolor},  
    commentstyle=\color{codegreen},     
    keywordstyle=\color{codeblue}\bfseries,  
    numberstyle=\tiny\color{codegray},  
    stringstyle=\color{red},            
    basicstyle=\ttfamily\small,        
    breakatwhitespace=false,           
    breaklines=true,                    
    captionpos=b,                       
    keepspaces=true,                    
    numbers=left,                       
    numbersep=5pt,                      
    showspaces=false,                   
    showstringspaces=false,             
    showtabs=false,                     
    tabsize=4,                          
    frame=single                        
}

\lstdefinestyle{bashstyle}{
  backgroundcolor=\color{black},        % 背景颜色为黑色
  basicstyle=\ttfamily\color{white},     % 文字为白色的等宽字体
  keywordstyle=\color{cyan},             % 关键字为青色
  commentstyle=\color{gray},             % 注释为灰色
  stringstyle=\color{yellow},            % 字符串为黄色
  numberstyle=\tiny\color{gray},         % 行号为灰色
  numbersep=5pt,                         % 行号与代码间的距离
  frame=single,                          % 单线框架
  rulecolor=\color{white},               % 边框颜色为白色
  breaklines=true,                       % 自动换行
  postbreak=\mbox{\textcolor{red}{$\hookrightarrow$}\space}, % 换行时的箭头
  showstringspaces=false,                % 不显示字符串中的空格
  captionpos=b,                          % 标题位置为底部
  xleftmargin=20pt,                      % 左边距
  xrightmargin=20pt,                     % 右边距
  aboveskip=10pt,                        % 上方间距
  belowskip=10pt,                        % 下方间距
  numbers = none                         % 禁用编号
}

% 输入整个Python文件到Listing中
\newcommand{\pythoncode}[2][]{%
    \lstinputlisting[style=codestyle,language = python,#1]{#2}
}
\newcommand{\matlabcode}[2][]{
    \lstinputlisting[style=codestyle,language = matlab,#1]{#2}
}

% 预定义说明文字
\newcommand{\analyze}[1]{\CJKfamily{kai} 实验结果分析：#1\normalfont}
\newcommand{\figureAnalyze}[1]{\CJKfamily{kai} 图表分析： #1\normalfont}
\newcommand{\CodeReference}[1]{\CJKfamily{kai} 对应代码： #1 \normalfont}
% 文档开始
\endinput
