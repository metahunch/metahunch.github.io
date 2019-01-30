---
layout: page
title: Contributing an Article
---

If you're contributing an article, please read these instructions on how to write and post an article on this blog.

## Using Markdown

All articles are written in markdown with minimal formatting necessary. To a first approximation your article should simply be a plain text file. Here's the basic template your article should use:

```
---
layout:     post
title:      Using Markdown
date:       2019-01-30
author:     Your Name
comments:   false
---

Articles are written in markdown (kramdown).
Please use minimal formatting for your article.

This is a single paragraph. It is separated by newlines.
No need for html tags.
A single line break does not start a new paragraph.

This is a another paragraph.
You can place subsections as follows.

## Using Maths

You can use math as you would in latex. Let $n>2$ be an integer.
Consider the equation:
$$
a^n + b^n = c^n
$$

Wiles proved the following theorem.

> **Theorem** (Fermat's Last Theorem).
> The above equation has no solution in the positive integers.

## Links and Images

You can place a link like [this](http://wikipedia.org).

You place a picture similarly:

![Lander](https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Albert_Bierstadt_-_The_Rocky_Mountains%2C_Lander%27s_Peak.jpg/320px-Albert_Bierstadt_-_The_Rocky_Mountains%2C_Lander%27s_Peak.jpg)

## Emphasis and Boldface

Use *this* for emphasis and **this** for boldface.

## You're all set!

There really isn't more you need to know.
```

The post would appear like [this](/guide/example) on the web. If you need more than what is in the above example, check out this [kramdown reference](http://kramdown.gettalong.org/quickref.html). You may also use any valid HTML tag in your article, but please try to avoid this.


## Submitting an Article

The blog lives in this [github repository](https://github.com/metahunch/metahunch.github.io). If you are a regular contributor and your github account has admin access to the repository, you can add an article like this from your command line:

```
git clone https://github.com/metahunch/metahunch.github.io
cd metahunch.github.io
cp _drafts/2019-01-30-template.md _posts/year-mm-dd-your-post-tile.md
git add year-mm-dd-your-post-tile.md
# edit the article using your favorite editor
git commit -m ":pencil:"
git push -u origin master
```

If after that you make more changes you can upload them as follows:

```
git commit -am ":pencil:"
git pull
git push
```

If you don't have admin access, you can either create a [pull request](https://help.github.com/articles/creating-a-pull-request/) (assuming you're comfortable with git) or send any regular contributor the article in markdown. If so, please start from [this template](https://raw.githubusercontent.com/metahunch/metahunch.github.io/master/_drafts/2019-01-30-template.md) and follow the above.
