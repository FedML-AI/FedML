var jQuery = (typeof(window) != 'undefined') ? window.jQuery : require('jquery');

// Sphinx theme nav state
function ThemeNav () {

    var nav = {
        navBar: null,
        win: null,
        winScroll: false,
        winResize: false,
        linkScroll: false,
        winPosition: 0,
        winHeight: null,
        docHeight: null,
        isRunning: false
    };

    nav.enable = function (withStickyNav) {
        var self = this;

        // TODO this can likely be removed once the theme javascript is broken
        // out from the RTD assets. This just ensures old projects that are
        // calling `enable()` get the sticky menu on by default. All other cals
        // to `enable` should include an argument for enabling the sticky menu.
        if (typeof(withStickyNav) == 'undefined') {
            withStickyNav = true;
        }

        if (self.isRunning) {
            // Only allow enabling nav logic once
            return;
        }

        self.isRunning = true;
        jQuery(function ($) {
            self.init($);

            self.reset();
            self.win.on('hashchange', self.reset);

            if (withStickyNav) {
                // Set scroll monitor
                self.win.on('scroll', function () {
                    if (!self.linkScroll) {
                        if (!self.winScroll) {
                            self.winScroll = true;
                            requestAnimationFrame(function() { self.onScroll(); });
                        }
                    }
                });
            }

            // Set resize monitor
            self.win.on('resize', function () {
                if (!self.winResize) {
                    self.winResize = true;
                    requestAnimationFrame(function() { self.onResize(); });
                }
            });

            self.onResize();
        });

    };

    // TODO remove this with a split in theme and Read the Docs JS logic as
    // well, it's only here to support 0.3.0 installs of our theme.
    nav.enableSticky = function() {
        this.enable(true);
    };

    nav.init = function ($) {
        var doc = $(document),
            self = this;

        this.navBar = $('div.pytorch-side-scroll:first');
        this.win = $(window);

        // Set up javascript UX bits
        $(document)
            // Shift nav in mobile when clicking the menu.
            .on('click', "[data-toggle='pytorch-left-menu-nav-top']", function() {
                $("[data-toggle='wy-nav-shift']").toggleClass("shift");
                $("[data-toggle='rst-versions']").toggleClass("shift");
            })

            // Nav menu link click operations
            .on('click', ".pytorch-menu-vertical .current ul li a", function() {
                var target = $(this);
                // Close menu when you click a link.
                $("[data-toggle='wy-nav-shift']").removeClass("shift");
                $("[data-toggle='rst-versions']").toggleClass("shift");
                // Handle dynamic display of l3 and l4 nav lists
                self.toggleCurrent(target);
                self.hashChange();
            })
            .on('click', "[data-toggle='rst-current-version']", function() {
                $("[data-toggle='rst-versions']").toggleClass("shift-up");
            })

        // Make tables responsive
        $("table.docutils:not(.field-list,.footnote,.citation)")
            .wrap("<div class='wy-table-responsive'></div>");

        // Add extra class to responsive tables that contain
        // footnotes or citations so that we can target them for styling
        $("table.docutils.footnote")
            .wrap("<div class='wy-table-responsive footnote'></div>");
        $("table.docutils.citation")
            .wrap("<div class='wy-table-responsive citation'></div>");

        // Add expand links to all parents of nested ul
        $('.pytorch-menu-vertical ul').not('.simple').siblings('a').each(function () {
            var link = $(this);
                expand = $('<span class="toctree-expand"></span>');
            expand.on('click', function (ev) {
                self.toggleCurrent(link);
                ev.stopPropagation();
                return false;
            });
            link.prepend(expand);
        });
    };

    nav.reset = function () {
        // Get anchor from URL and open up nested nav
        var anchor = encodeURI(window.location.hash) || '#';

        try {
            var vmenu = $('.pytorch-menu-vertical');
            var link = vmenu.find('[href="' + anchor + '"]');
            if (link.length === 0) {
                // this link was not found in the sidebar.
                // Find associated id element, then its closest section
                // in the document and try with that one.
                var id_elt = $('.document [id="' + anchor.substring(1) + '"]');
                var closest_section = id_elt.closest('div.section');
                link = vmenu.find('[href="#' + closest_section.attr("id") + '"]');
                if (link.length === 0) {
                    // still not found in the sidebar. fall back to main section
                    link = vmenu.find('[href="#"]');
                }
            }
            // If we found a matching link then reset current and re-apply
            // otherwise retain the existing match
            if (link.length > 0) {
                $('.pytorch-menu-vertical .current').removeClass('current');
                link.addClass('current');
                link.closest('li.toctree-l1').addClass('current');
                link.closest('li.toctree-l1').parent().addClass('current');
                link.closest('li.toctree-l1').addClass('current');
                link.closest('li.toctree-l2').addClass('current');
                link.closest('li.toctree-l3').addClass('current');
                link.closest('li.toctree-l4').addClass('current');
            }
        }
        catch (err) {
            console.log("Error expanding nav for anchor", err);
        }

    };

    nav.onScroll = function () {
        this.winScroll = false;
        var newWinPosition = this.win.scrollTop(),
            winBottom = newWinPosition + this.winHeight,
            navPosition = this.navBar.scrollTop(),
            newNavPosition = navPosition + (newWinPosition - this.winPosition);
        if (newWinPosition < 0 || winBottom > this.docHeight) {
            return;
        }
        this.navBar.scrollTop(newNavPosition);
        this.winPosition = newWinPosition;
    };

    nav.onResize = function () {
        this.winResize = false;
        this.winHeight = this.win.height();
        this.docHeight = $(document).height();
    };

    nav.hashChange = function () {
        this.linkScroll = true;
        this.win.one('hashchange', function () {
            this.linkScroll = false;
        });
    };

    nav.toggleCurrent = function (elem) {
        var parent_li = elem.closest('li');
        parent_li.siblings('li.current').removeClass('current');
        parent_li.siblings().find('li.current').removeClass('current');
        parent_li.find('> ul li.current').removeClass('current');
        parent_li.toggleClass('current');
    }

    return nav;
};

module.exports.ThemeNav = ThemeNav();

if (typeof(window) != 'undefined') {
    window.SphinxRtdTheme = {
        Navigation: module.exports.ThemeNav,
        // TODO remove this once static assets are split up between the theme
        // and Read the Docs. For now, this patches 0.3.0 to be backwards
        // compatible with a pre-0.3.0 layout.html
        StickyNav: module.exports.ThemeNav,
    };
}


// requestAnimationFrame polyfill by Erik Möller. fixes from Paul Irish and Tino Zijdel
// https://gist.github.com/paulirish/1579671
// MIT license

(function() {
    var lastTime = 0;
    var vendors = ['ms', 'moz', 'webkit', 'o'];
    for(var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
        window.requestAnimationFrame = window[vendors[x]+'RequestAnimationFrame'];
        window.cancelAnimationFrame = window[vendors[x]+'CancelAnimationFrame']
                                   || window[vendors[x]+'CancelRequestAnimationFrame'];
    }

    if (!window.requestAnimationFrame)
        window.requestAnimationFrame = function(callback, element) {
            var currTime = new Date().getTime();
            var timeToCall = Math.max(0, 16 - (currTime - lastTime));
            var id = window.setTimeout(function() { callback(currTime + timeToCall); },
              timeToCall);
            lastTime = currTime + timeToCall;
            return id;
        };

    if (!window.cancelAnimationFrame)
        window.cancelAnimationFrame = function(id) {
            clearTimeout(id);
        };
}());

$(".sphx-glr-thumbcontainer").removeAttr("tooltip");
$("table").removeAttr("border");

// This code replaces the default sphinx gallery download buttons
// with the 3 download buttons at the top of the page

var downloadNote = $(".sphx-glr-download-link-note.admonition.note");
if (downloadNote.length >= 1) {
    var tutorialUrlArray = $("#tutorial-type").text().split('/');
        tutorialUrlArray[0] = tutorialUrlArray[0] + "_source"

    var githubLink = "https://github.com/pytorch/tutorials/blob/master/" + tutorialUrlArray.join("/") + ".py",
        notebookLink = $(".reference.download")[1].href,
        notebookDownloadPath = notebookLink.split('_downloads')[1],
        colabLink = "https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads" + notebookDownloadPath;

    $("#google-colab-link").wrap("<a href=" + colabLink + " data-behavior='call-to-action-event' data-response='Run in Google Colab' target='_blank'/>");
    $("#download-notebook-link").wrap("<a href=" + notebookLink + " data-behavior='call-to-action-event' data-response='Download Notebook'/>");
    $("#github-view-link").wrap("<a href=" + githubLink + " data-behavior='call-to-action-event' data-response='View on Github' target='_blank'/>");
} else {
    $(".pytorch-call-to-action-links").hide();
}

//This code handles the Expand/Hide toggle for the Docs/Tutorials left nav items

$(document).ready(function() {
  var caption = "#pytorch-left-menu p.caption";
  var collapseAdded = $(this).not("checked");
  var chevronRight = "<i class='fa-solid fa-chevron-right'></i>"
  var chevronDown = "<i class='fa-solid fa-chevron-down'></i>"
  $(caption).each(function () {
    var menuName = this.innerText.replace(/[^\w\s]/gi, "").trim();
    $(this).find("span").addClass("checked");
    if (collapseAdded && sessionStorage.getItem(menuName) !== "expand" || sessionStorage.getItem(menuName) == "collapse") {
      $(this.firstChild).after("<span class='expand-menu menu-item-decorator'>" + chevronRight + "  </span>");
      $(this.firstChild).after("<span class='hide-menu collapse menu-item-decorator'>" + chevronDown + "</span>");
      $(this).next("ul").hide();
      $("#pytorch-left-menu p.caption").next("ul").first().toggle();
    } else if (collapseAdded || sessionStorage.getItem(menuName) == "expand") {
      $(this.firstChild).after("<span class='expand-menu collapse menu-item-decorator'>" + chevronRight + "</span>");
      $(this.firstChild).after("<span class='hide-menu menu-item-decorator'>" + chevronDown + "</span>");
      $("#pytorch-left-menu p.caption").next("ul").first().toggle();
    }
  });
  $("#pytorch-left-menu p.caption").next("ul").first().toggle();

  $(".expand-menu").on("click", function () {
    $(this).prev(".hide-menu").toggle();
    $(this).parent().next("ul").toggle();
    var menuName = $(this).parent().text().replace(/[^\w\s]/gi, "").trim();
    if (sessionStorage.getItem(menuName) == "collapse") {
      sessionStorage.removeItem(menuName);
    }
    sessionStorage.setItem(menuName, "expand");
    toggleList(this);
  });

  $(".hide-menu").on("click", function () {
    $(this).next(".expand-menu").toggle();
    $(this).parent().next("ul").toggle();
    var menuName = $(this).parent().text().replace(/[^\w\s]/gi, "").trim();
    if (sessionStorage.getItem(menuName) == "expand") {
      sessionStorage.removeItem(menuName);
    }
    sessionStorage.setItem(menuName, "collapse");
    toggleList(this);
  });


  $("#pytorch-left-menu p.caption").on("click", function () {
    // pull out the name from sessionStorage (to persist across visits)
    var menuName = $(this).text().replace(/[^\w\s]/gi, "").trim();

    var expandedState = sessionStorage.getItem(menuName);
    if (expandedState == null) {
        sessionStorage.setItem(menuName, "expand");
        expandedState = 'expand';
    }

    var isExpanded = expandedState == 'expand';

    if (isExpanded) {
        // swap the arrows
        $(this).children(".hide-menu").toggle();
        $(this).children(".expand-menu").toggle();

        // show the list
        $(this).next("ul").toggle()

        sessionStorage.setItem(menuName, "collapse");
    }else {
        // swap the arrows
        $(this).children(".hide-menu").toggle();
        $(this).children(".expand-menu").toggle();

        // show the list
        $(this).next("ul").toggle()
        
        sessionStorage.setItem(menuName, "expand");
    }
  });

  function toggleList(menuCommand) {
    $(menuCommand).toggle();
  }
});

// Build an array from each tag that's present

var tagList = $(".tutorials-card-container").map(function() {
    return $(this).data("tags").split(",").map(function(item) {
        return item.trim();
      });
}).get();

function unique(value, index, self) {
      return self.indexOf(value) == index && value != ""
    }

// Only return unique tags

var tags = tagList.sort().filter(unique);

// Add filter buttons to the top of the page for each tag

function createTagMenu() {
    tags.forEach(function(item){
    $(".tutorial-filter-menu").append(" <div class='tutorial-filter filter-btn filter' data-tag='" + item + "'>" + item + "</div>")
  })
};

createTagMenu();

// Remove hyphens if they are present in the filter buttons

$(".tags").each(function(){
    var tags = $(this).text().split(",");
    tags.forEach(function(tag, i ) {
       tags[i] = tags[i].replaceAll('-', ' ')
    })
    $(this).html(tags.join(", "));
});

// Remove hyphens if they are present in the card body

$(".tutorial-filter").each(function(){
    var tag = $(this).text();
    $(this).html(tag.replaceAll('-', ' '))
})

// Remove any empty p tags that Sphinx adds

$("#tutorial-cards p").each(function(index, item) {
    if(!$(item).text().trim()) {
        $(item).remove();
    }
});

// Jump back to top on pagination click

$(document).on("click", ".page", function() {
    $('html, body').animate(
      {scrollTop: $("#dropdown-filter-tags").position().top},
      'slow'
    );
});

var link = $("a[href='intermediate/speech_command_recognition_with_torchaudio.html']");

if (link.text() == "SyntaxError") {
    console.log("There is an issue with the intermediate/speech_command_recognition_with_torchaudio.html menu item.");
    link.text("Speech Command Recognition with torchaudio");
}

$(".stars-outer > i").hover(function() {
    $(this).prevAll().addBack().toggleClass("fas star-fill");
});

$(".stars-outer > i").on("click", function() {
    $(this).prevAll().each(function() {
        $(this).addBack().addClass("fas star-fill");
    });

    $(".stars-outer > i").each(function() {
        $(this).unbind("mouseenter mouseleave").css({
            "pointer-events": "none"
        });
    });
})

$("#pytorch-side-scroll-right li a").on("click", function (e) {
  var href = $(this).attr("href");
  $('html, body').stop().animate({
    scrollTop: $(href).offset().top - 100
  }, 850);
  e.preventDefault;
});

var lastId,
  topMenu = $("#pytorch-side-scroll-right"),
  topMenuHeight = topMenu.outerHeight() + 1,
  // All sidenav items
  menuItems = topMenu.find("a"),
  // Anchors for menu items
  scrollItems = menuItems.map(function () {
    var item = $(this).attr("href");
    if (item.length) {
      return item;
    }
  });

$(window).scroll(function () {
  var fromTop = $(this).scrollTop() + topMenuHeight;
  var article = ".section";

  $(article).each(function (i) {
    var offsetScroll = $(this).offset().top - $(window).scrollTop();
    if (
      offsetScroll <= topMenuHeight + 200 &&
      offsetScroll >= topMenuHeight - 200 &&
      scrollItems[i] == "#" + $(this).attr("id") &&
      $(".hidden:visible")
    ) {
      $(menuItems).removeClass("side-scroll-highlight");
      $(menuItems[i]).addClass("side-scroll-highlight");
    }
  });
});
$("#pytorch-left-menu p.caption").next("ul").first().toggle();