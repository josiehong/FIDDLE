wget https://github.com/JosieHong/FIDDLE/releases/download/v2.0.0/fiddle_tcn_qtof.zip
unzip fiddle_tcn_qtof.zip
rm fiddle_tcn_qtof.zip

wget https://github.com/JosieHong/FIDDLE/releases/download/v2.0.0/fiddle_fdr_qtof.zip
unzip fiddle_fdr_qtof.zip
rm fiddle_fdr_qtof.zip

wget https://github.com/JosieHong/FIDDLE/releases/download/v2.0.0/fiddle_tcn_orbitrap.zip
unzip fiddle_tcn_orbitrap.zip
rm fiddle_tcn_orbitrap.zip

wget https://github.com/JosieHong/FIDDLE/releases/download/v2.0.0/fiddle_fdr_orbitrap.zip
unzip fiddle_fdr_orbitrap.zip
rm fiddle_fdr_orbitrap.zip

mkdir -p check_point
mv fiddle_tcn_qtof.pt fiddle_fdr_qtof.pt fiddle_tcn_orbitrap.pt fiddle_fdr_orbitrap.pt check_point/