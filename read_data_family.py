# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 12:09:32 2014

@author: skew
"""

def read_data_family():
    a=file("kinship.data")
    b=file("ok.txt","a")
    b.write("relation,person1,person2\n")
    l=a.readlines()
    for s in l:
        if len(s)>2:
            print s
            first=str.split(s,"(")
            relation=first[0]
            second=str.split(first[1],",")
            person1=second[0]
            third=str.split(second[1],")")
            person2=third[0].replace(" ","")
            b.write(relation+","+person1+","+person2+"\n")
    a.close()
    b.close()