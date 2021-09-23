copy tmp1.arg tmp0.arg
tex2bmp.exe tmp --keep  --gscmd=%2 --more=tmp0.arg
copy tmppng.arg tmp0.arg
tex2bmp.exe tmp --keep  --gscmd=%2 --more=tmp0.arg
%3 -f emf -dt tmp.ps tmp.emf
copy tmp2.bmp tmp.bmp
erase tmp2.bmp
copy tmp2.png tmp.png
erase tmp2.png