%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% ETH PhD-thesis class
%% --------------------
%% Rene Fehlmann, 1996
%% Update to \maketitle by Philipp Unterbrunner, 2012
%%
%% Modification of book.cls with
%%
%% - new headings
%% - new titlepage
%% - new bibliography
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% \CharacterTable
%% {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%% Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%% Digits        \0\1\2\3\4\5\6\7\8\9
%% Exclamation   \!     Double quote  \"     Hash (number) \#
%% Dollar        \$     Percent       \%     Ampersand     \&
%% Acute accent  \'     Left paren    \(     Right paren   \)
%% Asterisk      \*     Plus          \+     Comma         \,
%% Minus         \-     Point         \.     Solidus       \/
%% Colon         \:     Semicolon     \;     Less than     \<
%% Equals        \=     Greater than  \>     Question mark \?
%% Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%% Grave accent  \`     Left brace    \{     Vertical bar  \|
%% Right brace   \}     Tilde         \~}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ETHthesis}
[1996/11/22 v1.0x
Non-standard LaTeX document class]
\typeout{ETH PhD thesis class. (Rene Fehlmann)}
\def\es{}
\newcommand{\@ptsize}{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagetrue
\newif\if@openright
\newif\if@openbib
\@openbibfalse
\newif\if@mainmatter \@mainmattertrue
\DeclareOption{a4paper}
{\setlength\paperheight {297mm}%
  \setlength\paperwidth  {210mm}}
\DeclareOption{a5paper}
{\setlength\paperheight {210mm}%
  \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
{\setlength\paperheight {250mm}%
  \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
{\setlength\paperheight {11in}%
  \setlength\paperwidth  {8.5in}}
\DeclareOption{legalpaper}
{\setlength\paperheight {14in}%
  \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
{\setlength\paperheight {10.5in}%
  \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
{\setlength\@tempdima   {\paperheight}%
  \setlength\paperheight {\paperwidth}%
  \setlength\paperwidth  {\@tempdima}}
\DeclareOption{10pt}{\renewcommand{\@ptsize}{0}}
\DeclareOption{11pt}{\renewcommand{\@ptsize}{1}}
\DeclareOption{12pt}{\renewcommand{\@ptsize}{2}}
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\DeclareOption{titlepage}{\@titlepagetrue}
\DeclareOption{notitlepage}{\@titlepagefalse}
\DeclareOption{openright}{\@openrighttrue}
\DeclareOption{openany}{\@openrightfalse}
\DeclareOption{onecolumn}{\@twocolumnfalse}
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{\@openbibtrue}
\DeclareOption{english}{
  \newif\ifenglish \englishtrue
  \newif\ifgerman  \germanfalse
}
\DeclareOption{german}{
  \newif\ifenglish \englishfalse
  \newif\ifgerman  \germantrue
}
\ExecuteOptions{letterpaper,10pt,twoside,onecolumn,final,openright,english}
\ProcessOptions
\input{bk1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand{\baselinestretch}{}
\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand{\topfraction}{.7}
\setcounter{bottomnumber}{1}
\renewcommand{\bottomfraction}{.3}
\setcounter{totalnumber}{3}
\renewcommand{\textfraction}{.2}
\renewcommand{\floatpagefraction}{.5}
\setcounter{dbltopnumber}{2}
\renewcommand{\dbltopfraction}{.7}
\renewcommand{\dblfloatpagefraction}{.5}

\if@twoside
\def\ps@headings{%
  \let\@oddfoot\@empty\let\@evenfoot\@empty
  \def\@evenhead{\underline{\makebox[\textwidth][l]{\es\rm \thepage
        \vphantom{g}\hfill{\bf
          \leftmark}}}}%
  \def\@oddhead{\underline{\makebox[\textwidth][l]{\es \hbox{}
        {\bf\rightmark}\vphantom{g}\hfill\rm\thepage}}}%
  \let\@mkboth\markboth
  \def\chaptermark##1{%
    \markboth {%
      \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
      \@chapapp\ \thechapter. \ %
      \fi
      \fi
      \uppercase{##1}}{}}%
  \def\sectionmark##1{%
    \markright {%
      \ifnum \c@secnumdepth >\z@
      \thesection. \ %
      \fi
      \uppercase{##1}}}}
\else
\def\ps@headings{%
  \let\@oddfoot\@empty
  \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
  \let\@mkboth\markboth
  \def\chaptermark##1{%
    \markright {\uppercase{%
        \ifnum \c@secnumdepth >\m@ne
        \if@mainmatter
        \@chapapp\ \thechapter. \ %
        \fi
        \fi
        ##1}}}}
\fi


\def\ps@myheadings{%
  \let\@oddfoot\@empty\let\@evenfoot\@empty
  \def\@evenhead{\thepage\hfil\slshape\leftmark}%
  \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
  \let\@mkboth\@gobbletwo
  \let\chaptermark\@gobble
  \let\sectionmark\@gobble
}
\if@titlepage
\newcommand{\maketitle}{
  \ifenglish
  \begin{titlepage}
    \setlength{\baselineskip}{8mm}
    \begin{center}
      \large DISS.\ ETH N$^\circ$\ \dissnumstring
    \end{center}
    \vspace{.022\linewidth}
    \begin{center}
      {\def\\{\linebreak}
        {\bf \fontsize{20}{24}\selectfont \titlestring} % \par
      }
    \end{center}
    \vspace{.04\linewidth}
    \begin{center}
      \large A thesis submitted to attain the degree of \vspace{.015\linewidth}
      \linebreak
      {\fontsize{14}{20}\selectfont DOCTOR OF SCIENCES of ETH ZURICH}\vspace{8pt}
      \linebreak
      (Dr. sc. ETH Zurich)
    \end{center}
    \vspace{.05\linewidth}
    \begin{center}
      \large presented by\vspace{.05\linewidth} \linebreak
      \uppercase\expandafter{\authorstring}\vspace{.015\linewidth} \linebreak
      \acatitlestring\vspace{.01\linewidth} \linebreak
      born on \dateofbirthstring\vspace{.01\linewidth} \linebreak
      % citizen of \citizenstring
    \end{center}
    \vspace{.05\linewidth}
    % \vspace{0.5\fill}
    \begin{center}
      \large Accepted on the recommendation of\vspace{8pt} \linebreak
      \examinersstring
    \end{center}
    % \vspace{0.5\fill}
    \vspace{.05\linewidth}
    \begin{center}
      \large\number\yearstring
    \end{center}

    \@ifundefined{@support}{}{{\medskip\footnotesize\noindent\@support\par}}
    \@ifundefined{@disclaimer}{}{{\medskip\footnotesize\noindent\@disclaimer\par}}
    \cleardoublepage
  \end{titlepage}
  \fi
  \ifgerman
  \begin{titlepage}
    \setlength{\baselineskip}{8mm}
    \begin{center}
      \large DISS.\ ETH Nr.\ \dissnumstring
    \end{center}
    \vfill
    \begin{center}
      {\def\\{\linebreak}
        \huge\bf\titlestring}
    \end{center}
    \vfill
    \begin{center}
      \large ABHANDLUNG \linebreak
      zur Erlangung des Titels \linebreak
      \degreestring
    \end{center}
    \vspace{0.5\fill}
    \begin{center}
      \large der \linebreak
      ETH Z\"URICH
    \end{center}
    \vspace{0.5\fill}
    \begin{center}
      \large vorgelegt von \linebreak
      \uppercase\expandafter{\authorstring} \linebreak
      \acatitlestring \linebreak
      geboren am \dateofbirthstring \linebreak
      von \citizenstring
    \end{center}
    \vspace{0.5\fill}
    \begin{center}
      \large Angenommen auf Antrag von \linebreak
      \examinersstring
    \end{center}
    \vspace{0.5\fill}
    \begin{center}
      \large\number\year
    \end{center}
  \end{titlepage}
  \fi
}
\else
\newcommand{\maketitle}{\par
  \begingroup
  \renewcommand{\thefootnote}{\fnsymbol{footnote}}%
  \def\@makefnmark{\hbox to\z@{$\m@th^{\@thefnmark}$\hss}}%
  \long\def\@makefntext##1{\parindent 1em\noindent
    \hbox to1.8em{\hss$\m@th^{\@thefnmark}$}##1}%
  \if@twocolumn
  \ifnum \col@number=\@ne
  \@maketitle
  \else
  \twocolumn[\@maketitle]%
  \fi
  \else
  \newpage
  \global\@topnum\z@   % Prevents figures from going at top of page.
  \@maketitle
  \fi
  \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \let\thanks\relax
  \let\maketitle\relax\let\@maketitle\relax
  \gdef\@thanks{}\gdef\@author{}\gdef\@title{}}
\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em}
\fi
\newcommand*{\chaptermark}[1]{}
\setcounter{secnumdepth}{2}
\newcounter {part}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand{\thepart}         {\Roman{part}}
\renewcommand{\thechapter}      {\arabic{chapter}}
\renewcommand{\thesection}      {\thechapter.\arabic{section}}
\renewcommand{\thesubsection}   {\thesection.\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection .\arabic{subsubsection}}
\renewcommand{\theparagraph}    {\thesubsubsection.\arabic{paragraph}}
\renewcommand{\thesubparagraph} {\theparagraph.\arabic{subparagraph}}
\newcommand{\@chapapp}{\chaptername}
\newcommand{\frontmatter}{\cleardoublepage
  \@mainmatterfalse\pagenumbering{roman}}
\newcommand{\mainmatter}{\cleardoublepage
  \@mainmattertrue\pagenumbering{arabic}}
\newcommand{\backmatter}{\if@openright\cleardoublepage\else\clearpage\fi
  \@mainmatterfalse}
\newcommand{\part}{\cleardoublepage
  \thispagestyle{plain}%
  \if@twocolumn
  \onecolumn
  \@tempswatrue
  \else
  \@tempswafalse
  \fi
  \hbox{}\vfil
  \secdef\@part\@spart}

\def\@part[#1]#2{%
  \ifnum \c@secnumdepth >-2\relax
  \refstepcounter{part}%
  \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
  \else
  \addcontentsline{toc}{part}{#1}%
  \fi
  \markboth{}{}
  {\centering
    \interlinepenalty \@M
    \reset@font
    \ifnum \c@secnumdepth >-2\relax
    \huge\bfseries \partname~\thepart
    \par
    \vskip 20\p@
    \fi
    \Huge \bfseries #2\par}%
  \@endpart}
\def\@spart#1{%
  {\centering
    \interlinepenalty \@M
    \reset@font
    \Huge \bfseries #1\par}%
  \@endpart}
\def\@endpart{\vfil\newpage
  \if@twoside
  \hbox{}%
  \thispagestyle{empty}%
  \newpage
  \fi
  \if@tempswa
  \twocolumn
  \fi}
\newcommand{\chapter}{\if@openright\cleardoublepage\else\clearpage\fi
  \thispagestyle{plain}%
  \global\@topnum\z@
  \@afterindentfalse
  \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
  \if@mainmatter
  \refstepcounter{chapter}%
  \typeout{\@chapapp\space\thechapter.}%
  \addcontentsline{toc}{chapter}%
  {\protect\numberline{\thechapter}#1}%
  \else
  \addcontentsline{toc}{chapter}{#1}%
  \fi
  \else
  \addcontentsline{toc}{chapter}{#1}%
  \fi
  \chaptermark{#1}%
  \addtocontents{lof}{\protect\addvspace{10\p@}}%
  \addtocontents{lot}{\protect\addvspace{10\p@}}%
  \if@twocolumn
  \@topnewpage[\@makechapterhead{#2}]%
  \else
  \@makechapterhead{#2}%
  \@afterheading
  \fi}
\def\@makechapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright \reset@font
    \ifnum \c@secnumdepth >\m@ne
    \if@mainmatter
    \Large\bfseries \@chapapp{} \thechapter
    \par\nobreak
    \vskip 20\p@
    \fi
    \fi
    \interlinepenalty\@M
    \Huge \bfseries #1\par\nobreak
    \vskip 40\p@
  }}
\def\@schapter#1{\if@twocolumn
  \@topnewpage[\@makeschapterhead{#1}]%
  \else
  \@makeschapterhead{#1}%
  \@afterheading
  \fi}
\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \reset@font
    \interlinepenalty\@M
    \Huge \bfseries  #1\par\nobreak
    \vskip 40\p@
  }}
\newcommand{\section}{\@startsection {section}{1}{\z@}%
  {-3.5ex \@plus -1ex \@minus -.2ex}%
  {2.3ex \@plus.2ex}%
  {\reset@font\Large\bfseries}}
\newcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\reset@font\large\bfseries}}
\newcommand{\subsubsection}{\@startsection{subsubsection}{3}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\reset@font\normalsize\bfseries}}
\newcommand{\paragraph}{\@startsection{paragraph}{4}{\z@}%
  {3.25ex \@plus1ex \@minus.2ex}%
  {-1em}%
  {\reset@font\normalsize\bfseries}}
\newcommand{\subparagraph}{\@startsection{subparagraph}{5}{\parindent}%
  {3.25ex \@plus1ex \@minus .2ex}%
  {-1em}%
  {\reset@font\normalsize\bfseries}}
\if@twocolumn
\setlength\leftmargini  {2em}
\else
\setlength\leftmargini  {2.5em}
\fi
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
\setlength\leftmarginv  {.5em}
\setlength\leftmarginvi {.5em}
\else
\setlength\leftmarginv  {1em}
\setlength\leftmarginvi {1em}
\fi
\setlength\leftmargin    {\leftmargini}
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand{\theenumi}{\arabic{enumi}}
\renewcommand{\theenumii}{\alph{enumii}}
\renewcommand{\theenumiii}{\roman{enumiii}}
\renewcommand{\theenumiv}{\Alph{enumiv}}
\newcommand{\labelenumi}{\theenumi.}
\newcommand{\labelenumii}{(\theenumii)}
\newcommand{\labelenumiii}{\theenumiii.}
\newcommand{\labelenumiv}{\theenumiv.}
\renewcommand{\p@enumii}{\theenumi}
\renewcommand{\p@enumiii}{\theenumi(\theenumii)}
\renewcommand{\p@enumiv}{\p@enumiii\theenumiii}
\newcommand{\labelitemi}{$\m@th\bullet$}
\newcommand{\labelitemii}{\normalfont\bfseries --}
\newcommand{\labelitemiii}{$\m@th\ast$}
\newcommand{\labelitemiv}{$\m@th\cdot$}
\newenvironment{description}
{\list{}{\labelwidth\z@ \itemindent-\leftmargin
    \let\makelabel\descriptionlabel}}
{\endlist}
\newcommand{\descriptionlabel}[1]{\hspace\labelsep
  \normalfont\bfseries #1}
\newenvironment{verse}
{\let\\=\@centercr
  \list{}{\itemsep      \z@
    \itemindent   -1.5em%
    \listparindent\itemindent
    \rightmargin  \leftmargin
    \advance\leftmargin 1.5em}%
\item[]}
{\endlist}
\newenvironment{quotation}
{\list{}{\listparindent 1.5em%
    \itemindent    \listparindent
    \rightmargin   \leftmargin
    \parsep        \z@ \@plus\p@}%
\item[]}
{\endlist}
\newenvironment{quote}
{\list{}{\rightmargin\leftmargin}%
\item[]}
{\endlist}
\newenvironment{titlepage}
{%
  \cleardoublepage
  \if@twocolumn
  \@restonecoltrue\onecolumn
  \else
  \@restonecolfalse\newpage
  \fi
  \thispagestyle{empty}%
  \if@compatibility
  \setcounter{page}{0}
  \fi}%
{\if@restonecol\twocolumn \else \newpage \fi
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\nochapter}[1]{
  \chapter*{#1
    \@mkboth{\uppercase{#1}}{\uppercase{#1}}}
  \addcontentsline{toc}{chapter}
  {\protect\numberline{}#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\appendix}{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \renewcommand{\@chapapp}{\appendixname}%
  \renewcommand{\thechapter}{\Alph{chapter}}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\@addtoreset{equation}{chapter}
\renewcommand{\theequation}{\thechapter.\arabic{equation}}
\newcounter{figure}[chapter]
\renewcommand{\thefigure}{\thechapter.\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}
{\@float{figure}}
{\end@float}
\newenvironment{figure*}
{\@dblfloat{figure}}
{\end@dblfloat}
\newcounter{table}[chapter]
\renewcommand{\thetable}{\thechapter.\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
{\@float{table}}
{\end@float}
\newenvironment{table*}
{\@dblfloat{table}}
{\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1: #2}%
  \ifdim \wd\@tempboxa >\hsize
  #1: #2\par
  \else
  \global \@minipagefalse
  \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*{\cal}{\@fontswitch{\relax}{\mathcal}}
\DeclareRobustCommand*{\mit}{\@fontswitch{\relax}{\mathnormal}}
\newcommand{\@pnumwidth}{1.55em}
\newcommand{\@tocrmarg} {2.55em}
\newcommand{\@dotsep}{4.5}
\setcounter{tocdepth}{2}
\newcommand{\tableofcontents}{%
  \if@twocolumn
  \@restonecoltrue\onecolumn
  \else
  \@restonecolfalse
  \fi
  \chapter*{\contentsname
    \@mkboth{\uppercase{\contentsname}}{\uppercase{\contentsname}}}%
  \@starttoc{toc}%
  \if@restonecol\twocolumn\fi
}
\newcommand*{\l@part}[2]{%
  \ifnum \c@tocdepth >-2\relax
  \addpenalty{-\@highpenalty}%
  \addvspace{2.25em \@plus\p@}%
  \begingroup
  \setlength\@tempdima{3em}%
  \parindent \z@ \rightskip \@pnumwidth
  \parfillskip -\@pnumwidth
  {\leavevmode
    \large \bfseries #1\hfil \hbox to\@pnumwidth{\hss #2}}\par
  \nobreak
  \global\@nobreaktrue
  \everypar{\global\@nobreakfalse\everypar{}}
  \endgroup
  \fi}
\newcommand*{\l@chapter}[2]{%
  \ifnum \c@tocdepth >\m@ne
  \addpenalty{-\@highpenalty}%
  \vskip 1.0em \@plus\p@
  \setlength\@tempdima{1.5em}%
  \begingroup
  \parindent \z@ \rightskip \@pnumwidth
  \parfillskip -\@pnumwidth
  \leavevmode \bfseries
  \advance\leftskip\@tempdima
  \hskip -\leftskip
  #1\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss #2}\par
  \penalty\@highpenalty
  \endgroup
  \fi}
\newcommand*{\l@section}      {\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*{\l@subsection}   {\@dottedtocline{2}{3.8em}{3.2em}}
\newcommand*{\l@subsubsection}{\@dottedtocline{3}{7.0em}{4.1em}}
\newcommand*{\l@paragraph}    {\@dottedtocline{4}{10em}{5em}}
\newcommand*{\l@subparagraph} {\@dottedtocline{5}{12em}{6em}}
\newcommand{\listoffigures}{%
  \if@twocolumn
  \@restonecoltrue\onecolumn
  \else
  \@restonecolfalse
  \fi
  \chapter*{\listfigurename
    \@mkboth{\uppercase{\listfigurename}}%
    {\uppercase{\listfigurename}}}%
  \@starttoc{lof}%
  \if@restonecol\twocolumn\fi
}
\newcommand*{\l@figure}{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand{\listoftables}{%
  \if@twocolumn
  \@restonecoltrue\onecolumn
  \else
  \@restonecolfalse
  \fi
  \chapter*{\listtablename
    \@mkboth{\uppercase{\listtablename}}{\uppercase{\listtablename}}}%
  \@starttoc{lot}%
  \if@restonecol\twocolumn\fi
}
\let\l@table\l@figure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newdimen\bibindent
\bibindent=1.5em
% \newenvironment{thebibliography}[1]
% {\chapter*{\bibname
% \@mkboth{\uppercase{\bibname}}{\uppercase{\bibname}}}%
% \addcontentsline{toc}{chapter}
% {\protect\numberline{}\bibname}
% \list{}%
% { \labelwidth0ex \leftmargin2em \itemindent-2em \labelsep0ex
% \rightmargin0mm \itemsep1ex \topsep0ex \parsep0.2ex
% \partopsep0ex }
% \if@openbib
% \renewcommand{\newblock}{\par}
% \else
% \renewcommand{\newblock}{\hskip .11em \@plus.33em \@minus.07em}%
% \fi
% \sloppy\clubpenalty4000\widowpenalty4000%
% \sfcode`\.=\@m}
% {\def\@noitemerr
% {\@latex@warning{Empty `thebibliography' environment}}%
% \endlist}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newenvironment{thebibliography}[1]
{\chapter*{\bibname
    \@mkboth{\uppercase{\bibname}}{\uppercase{\bibname}}}%
  \addcontentsline{toc}{chapter}
  {\protect\numberline{}\bibname}
  \list{\@biblabel{\arabic{enumiv}}}%
  {\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \if@openbib
    \advance\leftmargin\bibindent
    \itemindent -\bibindent
    \listparindent \itemindent
    \parsep \z@
    \fi
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand{\theenumiv}{\arabic{enumiv}}}%
  \if@openbib
  \renewcommand{\newblock}{\par}
  \else
  \renewcommand{\newblock}{\hskip .11em \@plus.33em \@minus.07em}%
  \fi
  \sloppy\clubpenalty4000\widowpenalty4000%
  \sfcode`\.=\@m}
{\def\@noitemerr
  {\@latex@warning{Empty `thebibliography' environment}}%
  \endlist}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\newblock}{}
\newenvironment{theindex}
{\if@twocolumn
  \@restonecolfalse
  \else
  \@restonecoltrue
  \fi
  \columnseprule \z@
  \columnsep 35\p@
  \twocolumn[\@makeschapterhead{\indexname}]%
  \@mkboth{\uppercase{\indexname}}%
  {\uppercase{\indexname}}%
  \thispagestyle{plain}\parindent\z@
  \parskip\z@ \@plus .3\p@\relax
  \let\item\@idxitem}
{\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand{\@idxitem}  {\par\hangindent 40\p@}
\newcommand{\subitem}   {\par\hangindent 40\p@ \hspace*{20\p@}}
\newcommand{\subsubitem}{\par\hangindent 40\p@ \hspace*{30\p@}}
\newcommand{\indexspace}{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand{\footnoterule}{%
  \kern-3\p@
  \hrule width .4\columnwidth
  \kern 2.6\p@}
\@addtoreset{footnote}{chapter}
\long\def\@makefntext#1{%
  \parindent 1em%
  \noindent
  \hbox to 1.8em{\hss\@makefnmark}#1}
\newcommand{\contentsname}{Contents}
\newcommand{\listfigurename}{List of Figures}
\newcommand{\listtablename}{List of Tables}
\newcommand{\bibname}{References}
\newcommand{\indexname}{Index}
\newcommand{\figurename}{Fig.}
\newcommand{\tablename}{Table}
\newcommand{\partname}{Part}
\newcommand{\chaptername}{Chapter}
\newcommand{\appendixname}{Appendix}
\newcommand{\today}{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{headings}
\pagenumbering{arabic}      % Arabic page numbers

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\dissnum#1{\def\dissnumstring{#1}}
\def\dissnumstring{}
\def\title#1{\def\titlestring{#1}}
\def\titlestring{}
\def\degree#1{\def\degreestring{#1}}
\def\degreestring{}
\def\author#1{\def\authorstring{#1}}
\def\authorstring{}
\def\acatitle#1{\def\acatitlestring{#1}}
\def\acatitlestring{}
\def\dateofbirth#1{\def\dateofbirthstring{#1}}
\def\dateofbirthstring{}
\def\citizen#1{\def\citizenstring{#1}}
\def\citizenstring{}
\def\examiners#1{\def\examinersstring{#1}}
\def\examinersstring{}
\def\Year#1{\def\yearstring{#1}}
\def\yearstring{}
\def\support#1{\gdef\@support{#1}}
\def\disclaimer#1{\gdef\@disclaimer{#1}}


\if@twoside
\else
\raggedbottom
\fi
\if@twocolumn
\twocolumn
\sloppy
\flushbottom
\else
\onecolumn
\fi
\endinput


%%
%% End of file `ETHthesis.cls'.
