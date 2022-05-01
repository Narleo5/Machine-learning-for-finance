=================================================================
ML for finance
ROBUST PRICING AND HEDGING VIA NEURAL SDES
Casimir Anatole - Jacquemet César - Thelot Léonard
=================================================================

.py :
	nsde_LV : A executer, entraine un réseau pour calibrer
		une surface de volatilité implicite, plot les
		résultats et la loss
	networks : Définit les réseaux appelés dans nsde_LV
	implied_vol : Retrouve la surface d'IV à partir des données
		      fournies par l'article

.csv :
- Prix de calls pour 6 strikes OTM et 6 maturités
sur index SPX et SX5E utilisées pour calibrer le réseau de neurones 
- Données disponibles ici  (Implied Volatility Surface by Moneyness):
https://www.ivolatility.com/data/data_download_intro.html
- Traitement des csv dans les notebooks du dossier "market data"

.pt :
Target data de l'article (prix générée par un modèle de Heston par les auteurs)

Folders :
	- market_data : Contient les notebooks et les csv pour avoir les
		      surfaces de vol du marhcé
	- plots : Contient des plots des loss, des IV et des prix, les réseaux entrainés, ...

