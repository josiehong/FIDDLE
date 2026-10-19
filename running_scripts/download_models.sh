wget https://github.com/JosieHong/FIDDLE/releases/download/v2.2.0/fiddle_tcn_qtof.zip
unzip fiddle_tcn_qtof.zip
rm fiddle_tcn_qtof.zip

wget https://github.com/JosieHong/FIDDLE/releases/download/v2.2.0/fiddle_rescore_qtof.zip
unzip fiddle_rescore_qtof.zip
rm fiddle_rescore_qtof.zip

wget https://github.com/JosieHong/FIDDLE/releases/download/v2.2.0/fiddle_tcn_orbitrap.zip
unzip fiddle_tcn_orbitrap.zip
rm fiddle_tcn_orbitrap.zip

wget https://github.com/JosieHong/FIDDLE/releases/download/v2.2.0/fiddle_rescore_orbitrap.zip
unzip fiddle_rescore_orbitrap.zip
rm fiddle_rescore_orbitrap.zip

mkdir -p check_point
mv fiddle_tcn_qtof.pt fiddle_rescore_qtof.pt fiddle_tcn_orbitrap.pt fiddle_rescore_orbitrap.pt check_point/