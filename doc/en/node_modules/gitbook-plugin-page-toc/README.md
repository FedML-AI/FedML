# gitbook-plugin-page-toc

This plugin adds a table of contents (TOC) to each page in your Gitbook.
You can set whether the TOC appears on all pages by default, and you can enable or disable the TOC on individual pages to override the default.

![](https://raw.githubusercontent.com/aleung/gitbook-plugin-page-toc/master/doc/screenshot-1.png)

## Install

Add the plugin to your `book.json`:

``` json
{
  "plugins": [ "page-toc" ],
  "pluginsConfig": {
    "page-toc": {
      "selector": ".markdown-section h1, .markdown-section h2, .markdown-section h3, .markdown-section h4",
      "position": "before-first",
      "showByDefault": true
    }
  }
}
```

## Configuration

- `selector` : CSS selector to select the elements to put anchors on
  - Default: `.markdown-section h1, .markdown-section h2, .markdown-section h3, .markdown-section h4`,
    which include headings from level 1 to level 4.
- `position` : Position of TOC
  - Allowed values:
    - `before-first` _(default)_ : Before the first heading
    - `top` : On top of the page
- `showByDefault`: Whether to show the TOC on all pages by default.
  - Default:  `true`.

## Use

To show a TOC in one of your pages, either set the `showByDefault` parameter to `true` in your `book.json`, or add the front matter item `showToc: true` to the top of the Markdown file like this:
```markdown
---
showToc: true
---
# My interesting page that has a TOC
```

If you have the `showByDefault` parameter set to `true` and you want to hide the TOC on a page, add the front matter item `showToc: false` to the top of the Markdown file like this:
```markdown
---
showToc: false
---
# My interesting page that does not have a TOC
```

The page-specific front matter overrides the `showByDefault` parameter.

## CSS Customization

The TOC elements have class attribute `.page-toc`. You can override the styles in `styles/website.css`.
