#!/bin/bash

echo Installing boost
sudo apt-get install libboost-all-dev

echo Installing Cifs
sudo apt-get install cifs-utils

echo Installing gedit 
sudo apt-get install gedit
git config --global user.email "apoorva.gupta@tum.de"
git config --global user.name "Apoorva Gupta"

echo Creating folder to mount mwn storage
mkdir lrz
sudo mount -t cifs //nas.ads.mwn.de/ga58qob ./lrz -o username=ga58qob,domain=ADS,forceuid,forcegid,uid=1000,gid=1000

echo files
cp lrz/madpet2.p016.csr4 .
cp lrz/Trues_Derenzo_GATE_rot_sm_200k.LMsino .






