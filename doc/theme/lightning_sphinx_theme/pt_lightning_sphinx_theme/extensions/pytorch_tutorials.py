"""Adapted from PyTorch Tutorials: https://github.com/pytorch/tutorials.

BSD 3-Clause License

Copyright (c) 2017, Pytorch contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import imp
from turtle import pd
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


class cardnode(nodes.General, nodes.TextElement):
    pass


class CustomCardItemDirective(SphinxDirective):
    option_spec = {
        "header": directives.unchanged,
        "image": directives.unchanged,
        "card_description": directives.unchanged,
        "tags": directives.unchanged,
        "beta": directives.flag,
    }

    def run(self):
        try:
            if "header" in self.options:
                header = self.options["header"]
            else:
                raise ValueError("header not doc found")

            if "image" in self.options:
                image = "<img src='" + self.options["image"] + "'>"
            else:
                image = "<img src='_static/images/icon.svg'>"

            # TODO: This probably only works when the tutorial list directive is in index.html
            link = self.env.docname + ".html"

            if "card_description" in self.options:
                card_description = self.options["card_description"]
            else:
                card_description = ""

            if "tags" in self.options:
                tags = self.options["tags"]
            else:
                tags = ""

            if "beta" in self.options:
                beta = "<span class='badge badge-secondary'>Beta</span>"
            else:
                beta = ""

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        card_rst = CARD_TEMPLATE.format(
            header=header, image=image, link=link, card_description=card_description, tags=tags, beta=beta,
        )
        card_list = StringList(card_rst.split("\n"))
        node = cardnode()
        self.state.nested_parse(card_list, self.content_offset, node)

        if not hasattr(self.env, "all_cardnodes"):
            self.env.all_cardnodes = []
        self.env.all_cardnodes.append({"docname": self.env.docname, "node": node})
        return [node]


CARD_TEMPLATE = """
.. raw:: html

    <div class="col-md-12 tutorials-card-container" data-tags={tags}>
        <div class="card tutorials-card">
            <a href="{link}">
                <div class="card-body">
                    <div class="card-title-container">
                        <h4>{header} {beta}</h4>
                    </div>
                    <p class="card-summary">{card_description}</p>
                    <p class="tags">{tags}</p>
                    <div class="tutorials-image">{image}</div>
                </div>
            </a>
        </div>
    </div>
"""


class CustomCalloutItemDirective(Directive):
    option_spec = {
        "header": directives.unchanged,
        "description": directives.unchanged,
        "button_link": directives.unchanged,
        "button_text": directives.unchanged,
        "col_css": directives.unchanged,
        "card_style": directives.unchanged,
        "image_center": directives.unchanged,
    }

    def run(self):
        try:
            if "description" in self.options:
                description = self.options["description"]
            else:
                description = ""

            if "header" in self.options:
                header = self.options["header"]
            else:
                raise ValueError("header not doc found")

            if "button_link" in self.options:
                button_link = self.options["button_link"]
            else:
                button_link = ""

            if "button_text" in self.options:
                button_text = self.options["button_text"]
            else:
                button_text = ""
            
            if "col_css" in self.options:
                col_css = self.options["col_css"]
            else:
                col_css = "col-md-6"
            
            if "card_style" in self.options:
                card_style = self.options["card_style"]
            else:
                card_style = "text-container"
            
            if "image_center" in self.options:
                image_center = "<img src='" + self.options["image_center"] + "'>"
            else:
                image_center = ""

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        callout_rst = CALLOUT_TEMPLATE.format(
            description=description, image_center=image_center, header=header, button_link=button_link, button_text=button_text, col_css=col_css, card_style=card_style
        )
        callout_list = StringList(callout_rst.split("\n"))
        callout = nodes.paragraph()
        self.state.nested_parse(callout_list, self.content_offset, callout)
        return [callout]


CALLOUT_TEMPLATE = """
.. raw:: html

    <div class="{col_css}">
        <a href="{button_link}">
            <div class="{card_style}">
                    <div class="image-center">{image_center}</div>
                    <h3>{header}</h3>
                    <p class="body-paragraph">{description}</p>
            </div>
        </a>
    </div>
"""



class DisplayItemDirective(Directive):
    option_spec = {
        "header": directives.unchanged,
        "description": directives.unchanged,
        "col_css": directives.unchanged,
        "card_style": directives.unchanged,
        "image_center": directives.unchanged,
        "image_right": directives.unchanged,
        "image_height": directives.unchanged,
        "button_link": directives.unchanged,
        "height": directives.unchanged,
        "tag": directives.unchanged,
    }

    def run(self):
        try:
            if "description" in self.options:
                description = self.options["description"]
            else:
                description = ""
            
            if "tag" in self.options:
                tag = "<div class='card-tag'>" + self.options["tag"] + "</div>"
            else:
                tag = ""

            if "height" in self.options:
                height = self.options["height"]
            else:
                height = ""

            if "header" in self.options:
                header = self.options["header"]
            else:
                raise ValueError("header not doc found")
 
            if "col_css" in self.options:
                col_css = self.options["col_css"]
            else:
                col_css = "col-md-6"
            
            if "card_style" in self.options:
                card_style = self.options["card_style"]
            else:
                card_style = "display-card"
            
            if "image_height" in self.options:
                image_height = self.options["image_height"]
            else:
                image_height = "125px"
            
            image_class = ''
            if "image_center" in self.options:
                image = "<img src='" + self.options["image_center"] + "' style=height:" + image_height + "  >"
                image_class = 'image-center'
            
            elif "image_right" in self.options:
                image = "<img src='" + self.options["image_right"] + "' style=height:" + image_height + "  >"
                image_class = 'image-right'
            else:
                image = ""
            
            if "button_link" in self.options:
                button_link = self.options["button_link"]
                button_open_html = f"<a href='{button_link}'>"
                button_close_html = "</a>"
                card_style = f'{card_style} display-card-hover'
            else:
                button_link = ""
                button_open_html = ""
                button_close_html = ""

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        callout_rst = DISPLAY_ITEM_TEMPLATE.format(
            description=description, 
            image=image, 
            height=height, 
            image_class=image_class,
            header=header, 
            col_css=col_css, 
            card_style=card_style,
            button_open_html=button_open_html,
            button_close_html=button_close_html,
            tag=tag
        )
        callout_list = StringList(callout_rst.split("\n"))
        callout = nodes.paragraph()
        self.state.nested_parse(callout_list, self.content_offset, callout)
        return [callout]


DISPLAY_ITEM_TEMPLATE = """
.. raw:: html

    <div class="{col_css}">
        {button_open_html}
        <div class="{card_style}" style='height:{height}px'>
                <div class="{image_class}">{image}</div>
                <h3>{header}</h3>
                <p class="body-paragraph">{description}</p>
        </div>
        {button_close_html}
        {tag}
    </div>
"""


class SlackButton(Directive):
    option_spec = {
        "align": directives.unchanged,
        "title": directives.unchanged,
        "width": directives.unchanged,
        "margin": directives.unchanged,
    }


    def run(self):
        try:
            # button width
            width = '155';
            if "width" in self.options:
                width = self.options["width"]
            
            # margin
            margin = '40';
            if "margin" in self.options:
                margin = self.options["margin"]
            
            # title on button
            title = 'Join our community'
            if "title" in self.options:
                title = self.options["title"]

            # button on left, center or right of screen 
            align = 'left'
            if "align" in self.options:
                align = self.options["align"]

            align = f'slack-align-{align}'

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []
        callout_rst = SLACK_TEMPLATE.format(
            align=align,
            title=title,
            margin=margin,
            width=width
        )
        callout_list = StringList(callout_rst.split("\n"))
        callout = nodes.paragraph()
        self.state.nested_parse(callout_list, self.content_offset, callout)
        return [callout]


SLACK_TEMPLATE = """
.. raw:: html

    <div class='row slack-container {align}' style="margin: {margin}px 0 {margin}px 0;"> 
        <div class='slack-button' style='width: {width}px'>
            <a href="https://join.slack.com/t/pytorch-lightning/shared_invite/zt-12iz3cds1-uyyyBYJLiaL2bqVmMN7n~A" target="_blank">
            <div class="icon" data-icon="slackLogo">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313zM8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zM8.834 6.313a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312zM18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zM17.688 8.834a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312zM15.165 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zM15.165 17.688a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z" fill="currentColor">
                    </path>
                </svg>
            </div>
            <span class='button-title'>{title}</span>
            </a>
        </div>
    </div>
"""


class TwoColumns(Directive):
    has_content = True
    option_spec = {
        "left": directives.unchanged,
        "right": directives.unchanged,
    }


    def run(self):
        try:
            left = ''
            if "left" in self.options:
                left = self.options["left"]
            
            right = ''
            if "right" in self.options:
                right = self.options["right"]
            
        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []
        callout_rst = TWO_COLUMN_TEMPLATE.format(
            left=left,
            right=right
        )
        callout_list = StringList(callout_rst.split("\n"))
        callout = nodes.paragraph()
        self.state.nested_parse(callout_list, self.content_offset, callout)
        return [callout]


TWO_COLUMN_TEMPLATE = """
.. raw:: html

   <div class="row" style='font-size: 14px; margin-bottom: 50px'>
      <div class='col-md-6'>

{left}

.. raw:: html

      </div>
      <div class='col-md-6'>

{right}

.. raw:: html

      </div>
   </div>
"""
