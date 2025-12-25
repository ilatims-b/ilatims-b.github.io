---
layout: default
title: Blog
permalink: /blog/
---

<ul class="post-list">
    {% for post in site.posts %}
    <li class="post-item">
        <span class="post-date">{{ post.date | date: "%Y-%m-%d" }}</span>
        <a class="post-title" href="{{ post.url }}">{{ post.title }}</a>
    </li>
    {% endfor %}
</ul>
