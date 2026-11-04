\NeedsTeXFormat{LaTeX2e} \ProvidesClass{template}
% Adapted from the RGU School of Computing report template (latex_template/RGU_Thesis_Template_2026):
% same page geometry, spacing and front-matter mechanics; the title page names
% the home institution and the declaration page is not used.

\RequirePackage{setspace}

\LoadClass[11pt, a4paper]{report}

\oddsidemargin 14.6truemm \evensidemargin 0truemm
\marginparwidth 40pt \marginparsep 10pt
\topmargin 0truemm \headsep 10truemm
\textheight 224truemm \textwidth 145truemm

\brokenpenalty=10000

\onehalfspacing

\def\degreetitle#1{\gdef\@degreetitle{#1}}
\def\rpttype#1{\gdef\@rpttype{#1}}
\def\principaladviser#1{\gdef\@principaladviser{#1}}
\def\advis@r{Adviser}
\def\principaladvisor#1{\gdef\@principaladviser{#1}\gdef\advis@r{Advisor}}
\def\firstreader#1{\gdef\@firstreader{#1}}
\def\secondreader#1{\gdef\@secondreader{#1}}
\def\submitdate#1{\gdef\@submitdate{#1}}
\def\copyrightyear#1{\gdef\@copyrightyear{#1}}
\def\institution#1{\gdef\@institution{#1}}
\def\@title{}\def\@author{}\def\@dept{}
\def\@principaladviser{}\def\@firstreader{}\def\@secondreader{}\def\@institution{}
\def\@submitdate{\ifcase\the\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space \number\the\year}

\newif\ifcopyright \newif\iffigurespage \newif\iftablespage \newif\ifalgorithmspage \newif\ifsigpage
\copyrightfalse \figurespagetrue \tablespagetrue \algorithmspagefalse \sigpagefalse

\def\titlep{%
        \thispagestyle{empty}%
        \null\vskip0.6in%
        \begin{center}
                \sc\expandafter{\@title}\\
        \vskip 0.35in
                {\large \@author}
        \vskip 0.45in
                {\rule{0.6\textwidth}{0.4pt}}
        \vfill
        \end{center}
        \begin{center}
                \sc \@rpttype\\[2pt] \@degreetitle\\[10pt]
                \normalfont\@institution
        \end{center}
        \vfill
        \begin{center}
                \@submitdate
        \end{center}
        \vskip.5in\newpage}

\def\beforeabstract{
        \pagenumbering{roman}
        \pagestyle{plain}
        \titlep}

\def\afterabstract{
        \newpage
\tableofcontents
        \newpage
        \iftablespage
                \addvspace{10pt}
                \listoftables
                \newpage
        \fi
        \iffigurespage
                \addvspace{10pt}
                \listoffigures
                \newpage
        \fi
    \ifalgorithmspage
                \addvspace{10pt}
                \listofalgorithms
                \newpage
        \fi}

\def\prefacesection#1{%
        \chapter*{#1}
        \addcontentsline{toc}{chapter}{#1}}

\def\afterpreface{\newpage
        \pagestyle{plain}}

\pagestyle{plain}
