# Update record
## v 1.0.14 - 2019-01-16
- [#38](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/38) Merge this pr to add the following functions
- fix bug : When the h1, h2, and h3 in md are in an abnormal order, the toc is missing. After the repair, the toc display will not be missed. It can be seen that the title is abnormal. V1.0.14+
- fix bug : In the previous fix bug, the serial number display in the case of showLevel was not taken into account.

## v 1.0.12 - 2018-09-17
- [#36](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/36) Merge this pr to add the following functions
- config.float.floatIcon ，You can configure the float navigation float icon style  V1.0.12+
- Add the `<!-- ex_nolevel -->` tag to the page，No hierarchical Numbers are generated on this page V1.0.12+

## v 1.0.11 - 2018-08-22
- [#33](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/33) The return top button is separated, and the return top button is displayed for true

## v1.0.9 - 2017-08-03
- fix bug: [#26](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/26) When configuring mode:, the title of the page is not rewritten

## v1.0.8 - 2017-08-03
- fix bug: [#26](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/26) mode:""
The configuration is invalid and the navigation is generated at the top of the page

## v1.0.5 - 2017-07-14
- New: the printLog configuration option, if it is true, compiles the printed MD file path at compile time. If the process fails, it is good to know which file it is
- New: multipleH1 configuration options, if true, according to a MD file with multiple H1 title, false, according to a MD file only contains a H1 title, the biggest difference is to remove the ugly 1.xxx in 1.

## v1.0.4 - 2017-06-02
This update is mainly to fix the title after the repeated strategy, and fix the previous version only dealt with the h1-h3 title duplicate bug[#19](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/19)
- Modify
     - Modify ID generation method
     - simplify the Toc function, optimize the code to speed up the running speed
- remember
     - Priority use of title content as ID
     - Keep custom ID [#18](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/18)
         - repeat ID auto-increment suffix [#6](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/6)
- Use the `<! - ex_nonav ->` annotation to let the page do not display navigation [#15](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/15)
## v1.0.2 - 2017-05-15
- Adding an `<extoc></extoc>` tag to the page generates the TOC directory here (which is consistent with the mode: pageTop pattern for the moment)[#17](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/17)

## v1.0.0 - 2017-03-09
- [#7](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/7)
- [#8](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/8)
- [#9](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/9)
- [#10](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/10)
- [#11](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/11)
- [#12](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/12)
- [#13](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/13)
- [#14](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/14)

#### v0.3.2 - 2017-03-08
- New configuration options - whether the page number is associated with the serial number generated in the official SUMMARY

#### v0.3.0 - 2017-03-06
- According to official level is associated with a page display [#4](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/4)

#### v0.2.7 - 2017-03-01
- fix bug: Anchor link index unique [#6](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/6)

#### v0.2.6 - 2017-03-01
- fix bug: [#5](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/5)

#### v0.2.5 - 2017-02-17
* to further optimize the suspension navigation style, and the official theme of the default to maintain the same, more beautiful, and unified style
* increase the suspension navigation TOC icon icon before the custom

#### v0.1.9 - 2017-02-17
Optimized navigation style

* reduce the shadow, slightly transparent panel background
* text display is 14 PX
* title number in bold
* adapt to the official default theme of the 3 sets. The navigation style will change with the style of the theme of the skin

#### v0.1.8 - 2017-02-09
* change the anchor icon display, the replacement for the CSS style. Otherwise, the PDF will fail to generate

#### v0.1.7 - 2017-02-09
* CSS named refactoring
* change the anchor icon display, the replacement for the GitHub consistent SVG Icon
* the generated directory is added to the top of the page and, in some cases, a navigation at the bottom of the page. Very unsightly, such as:
When the gitbook home page because it will not load the plug-in CSS effect
- CSS cannot be loaded while generating pdf

#### 2017-02-08
* rebuild project structure

#### 2017-02-07
* in the source code using the let and ES6 syntax, the use of OK in local, reported in gitbook: PluginError: Error with plugin "anchor-navigation-ex": Block-scoped declarations (let, const, function, class) not yet supported outside strict mode. Do not know why, or to VaR to declare it

#### 2017-02-06
* completely rewriting code
* with anchor and suspended navigation effect, now only need to introduce a plug-in gitbook-plugin-anchor-navigation-ex

#### 2017-01-18
* page without h[1-3] tag generation failed

#### 2017-01-22
* 2017-01-18 submitted a problem. Re repair
