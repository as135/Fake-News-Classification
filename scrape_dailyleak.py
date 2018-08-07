#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Author: Kelly Geyer, klg2@rice.edu
Date: May 31, 2017

Description: Script for collecting articles from http://www.dailyleak.org
'''


import os, json, uuid, re
import datetime
import urllib2
from bs4 import BeautifulSoup


def format_dailyleak_times(time_str): 
    '''
    This function formats the DailyLeak time str (technically unicode) into the RTG-FakeNews Format. Assumes UTC time
    zone.

    :param time_str: Time string formatted as u'June 11, 2014 4:49 pm'
    :return new_time_str: Time string formatted as 'YYYY-MM-DDTHH:MM:SSZ'
    '''
    #int_time = datetime.datetime.strptime(time_str, "%B %d, %Y %-I:%M %p")
    int_time = datetime.datetime.strptime(time_str, "%B %d, %Y %I:%M %p")
    new_time_str = unicode(int_time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return new_time_str


def parse_dailyleak_article(soup):
    '''
    This function parses an article from the DailyLeak, provided the soup object

    :param soup:
    :return dict: Returns dictionary containing the fields 'author', 'publishedAt', and 'article'.
    '''
    q = list(soup.children)[2]
    q1 = list(q.children)[1]
    # Get title of article
    txt_title =  list(q1.children)[11].get_text()
    q2 = list(q.children)[3]
    q3 = list(list(q2)[1])
    q4 = list(q3[1])
    raw = q4[4].get_text()
    # Get author
    author = u'Daily Leak'      # Same for all artiles
    # author = re.search('\r\n\n\n\n\n\n\n\nAuthor:(.*)\n\n\n\n\n\n\n\n\n\n\n\n\t/*', raw)
    # print author
    # try:
    #     author = author.group(1)
    #     author = author.strip()
    #     print author
    # except AttributeError:
    #     print "Can't find author of article"
    #     author = u''
    # Get date of publication
    dop = re.search('\n\n\n(.*)\n\n\n\n\n\n\n\r\n', raw)
    dop = dop.group(1)
    dop = dop.strip()
    # Convert to RTG-FakeNews date convention
    new_dop = format_dailyleak_times(dop)
    # Get article text
    art_text = u''
    txt_obj = list(list(list(list(q4[4].children)[1])[1])[6])
    for jj in txt_obj:
        if unicode(jj).startswith(u'<p>'):
            art_text += jj.get_text()
    return {u'author': author, u'publishedAt': dop, u'article': art_text}


def main():

    ##
    # Parameters
    main_dir = '/Users/kellygeyer/Documents/rtg_summer_2017'        # Main dir
    save_dir = os.path.join(main_dir, 'sample_data')                # Where you save clean data

    ##
    # Set up
    #
    # Website information
    main_url = 'http://www.dailyleak.org/category/'
    publisher = u'Daily Leak'
    # Sub-pages
    categories = ['entertainment', 'sports', 'media', 'world', 'health', 'politics', 'uncategorized']
    # Fake browser header to look less like a robot
    hdr  = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

    ##
    # Collect and parse HTML
    #
    # The desired fields of data are:
    # [u'publisher',        DONE
    #  u'description',      DONE
    #  u'author',           DONE
    #  u'url',              DONE
    #  u'title',            DONE
    #  u'publishedAt',      DONE
    #  u'urlToImage',       DONE
    #  u'article',          DONE
    #  u'dateOfCollection'  DONE
    #
    # For each page of site (see categories), collect URLs of articles
    art_objs = []       # article objects
    for cc in categories:
        print "\nCollect all URLs from the page", cc
        # Get HTML content of page
        url = main_url + cc + '/'
        req = urllib2.Request(url, headers=hdr)
        response = urllib2.urlopen(req)
        webContent = response.read()                        # HTML content
        soup = BeautifulSoup(webContent, 'html.parser')     # Convert it to soup object
        # Pull out links
        q1 = list(soup.children)[2]
        q5 = list(q1.children)[3]
        q6 = list(q5.children)[1]
        qq = q6.find_all(class_='entry-title')
        art_objs.extend(qq)

    # Get article from each link
    clean_dat = []
    for ii in art_objs:
        # Get article URL
        art_url = re.search('<a href="(.*)/"', str(ii))
        art_url = art_url.group(1)
        print "\nCollect article metadata from ", art_url
        # Get article title
        art_title = re.search('rel="bookmark">(.*)</a></h3>',str(ii))
        art_title = art_title.group(1)
        # Get article HTML
        req = urllib2.Request(art_url, headers=hdr)
        response = urllib2.urlopen(req)
        webContent = response.read()                        # HTML content
        soup = BeautifulSoup(webContent, 'html.parser')     # Convert it to soup object
        # Aggregate metadata
        art_dat = parse_dailyleak_article(soup)             # Metadata dictionary for article: author, article, publishedAt
        art_dat[u'url'] = art_url.strip()
        art_dat[u'title'] = art_title.strip()
        art_dat[u'publisher'] = publisher                   # Same publisher for all
        art_dat[u'description'] = u''                       # No description provided by site
        art_dat[u'urlToImage'] = u''                        # Not provided/collected
        now = datetime.datetime.utcnow()                    # Use UTC time zone ONLY
        art_dat[u'dateOfCollection'] = unicode(now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        clean_dat.append(art_dat)
        del art_dat

    # Save data
    print "Last step, save data!"
    uid = str(uuid.uuid4())
    clean_fn = os.path.join(save_dir, "dailyleak_" + uid + ".json")
    with open(clean_fn, "w") as json_file:
        json.dump(clean_dat, json_file)

    # OLD APPROACH
    # # Make request and download html
    # req = urllib2.Request(main_url, headers=hdr)
    # response = urllib2.urlopen(req)
    # webContent = response.read()                    # HTML content
    # soup = BeautifulSoup(webContent, 'html.parser')  # Convert it to soup object
    #
    # # Find article text in soup
    # q1 = list(soup.children)
    # q1_type = [type(item) for item in list(soup.children)]
    # for ii in q1:
    #     print ii
    #     if str(type(ii)) == "<class 'bs4.element.NavigableString'>":
    #         print "\nNavigableString: "
    #         print ii
    #     elif str(type(ii)) == "<class 'bs4.element.Comment'>":
    #         print "\nComment: "
    #         print ii
    #     elif str(type(ii)) == "<class 'bs4.element.Tag'>":
    #         print "\nComment: "
    #         print ii
    #     else:
    #         pass



if __name__ == '__main__':
    main()