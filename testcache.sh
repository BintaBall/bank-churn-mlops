# Test avec répétition des mêmes features
API_URL="https://bank-churn.salmonforest-247a5473.italynorth.azurecontainerapps.io"

# Premier appel (cache miss)
echo "Premier appel:"
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"CreditScore":650,"Age":42,"Tenure":5,"Balance":12500.50,"NumOfProducts":2,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":45000.00,"Geography_Germany":0,"Geography_Spain":1}' \
  | python3 -m json.tool

# Deuxième appel (cache hit)
echo -e "\nDeuxième appel (mêmes features):"
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"CreditScore":650,"Age":42,"Tenure":5,"Balance":12500.50,"NumOfProducts":2,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":45000.00,"Geography_Germany":0,"Geography_Spain":1}' \
  | python3 -m json.tool

# Voir les stats du cache
echo -e "\nStats du cache:"
curl -s "$API_URL/cache/stats" | python -m json.tool

# Test batch avec cache
echo -e "\nTest batch:"
curl -s -X POST "$API_URL/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[{"CreditScore":650,"Age":42,"Tenure":5,"Balance":12500.50,"NumOfProducts":2,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":45000.00,"Geography_Germany":0,"Geography_Spain":1},{"CreditScore":650,"Age":42,"Tenure":5,"Balance":12500.50,"NumOfProducts":2,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":45000.00,"Geography_Germany":0,"Geography_Spain":1}]' \
  | python3 -m json.tool