#!/bin/bash
set -euo pipefail

# Variables (à adapter)
RESOURCE_GROUP="rg-mlops"
LOCATION="italynorth"  # ou ta région
CONTAINER_APP_NAME="bank-churn"
APPINSIGHTS_NAME="bank-churn-insights"

echo "🔧 Configuration de Azure Application Insights..."

# 1. Vérification des prérequis
echo "1. Vérification des prérequis..."
az group show --name "$RESOURCE_GROUP" >/dev/null || {
    echo "❌ Resource Group $RESOURCE_GROUP introuvable"
    exit 1
}

# 2. Création d'Application Insights
echo "2. Création d'Application Insights: $APPINSIGHTS_NAME..."
az monitor app-insights component create \
  --app "$APPINSIGHTS_NAME" \
  --location "$LOCATION" \
  --resource-group "$RESOURCE_GROUP" \
  --application-type web \
  --query "{Name:name, AppId:appId, ConnectionString:connectionString}" \
  --output json > appinsights.json

echo "✅ Application Insights créé"

# 3. Récupération de la connection string
echo "3. Récupération de la connection string..."
APPINSIGHTS_CONN=$(jq -r '.ConnectionString' appinsights.json)

if [ -z "$APPINSIGHTS_CONN" ] || [ "$APPINSIGHTS_CONN" = "null" ]; then
    echo "❌ Impossible de récupérer la connection string"
    exit 1
fi

# Masque partiellement la clé pour l'affichage
MASKED_CONN=$(echo "$APPINSIGHTS_CONN" | sed 's/InstrumentationKey=[^;]*/InstrumentationKey=***/')
echo "Connection String: $MASKED_CONN"

# 4. Injection dans Azure Container Apps
echo "4. Configuration de Azure Container App..."
az containerapp update \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --set-env-vars "APPLICATIONINSIGHTS_CONNECTION_STRING=$APPINSIGHTS_CONN" \
  --query "{Name:name, EnvironmentVariables:properties.template.containers[0].env}" \
  --output json > containerapp_updated.json

echo "✅ Variables d'environnement mises à jour"

# 5. Vérification
echo "5. Vérification..."
az containerapp show \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "properties.template.containers[0].env[?name=='APPLICATIONINSIGHTS_CONNECTION_STRING'].value" \
  --output tsv | grep -q "InstrumentationKey" && \
  echo "✅ Application Insights configuré avec succès" || \
  echo "❌ Erreur de configuration"

# 6. Redémarrage pour prise en compte
echo "6. Redémarrage du Container App..."
az containerapp restart \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --no-wait

echo "⏳ Container App en cours de redémarrage..."

# 7. Informations utiles
APP_ID=$(jq -r '.AppId' appinsights.json)
echo ""
echo "=========================================="
echo "🎉 APPLICATION INSIGHTS CONFIGURÉ !"
echo "=========================================="
echo ""
echo "📊 Accès au monitoring :"
echo "   Portail Azure : https://portal.azure.com/#resource/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/microsoft.insights/components/$APPINSIGHTS_NAME/overview"
echo ""
echo "🔗 Application Insights ID : $APP_ID"
echo "📍 Région : $LOCATION"
echo ""
echo "🐳 Container App : $CONTAINER_APP_NAME"
echo "   Vérifie les logs dans 2-3 minutes :"
echo "   az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --follow"
echo ""
echo "🧪 Test du monitoring :"
echo "   curl https://$(az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)/health"
echo "=========================================="

# Nettoyage
rm -f appinsights.json containerapp_updated.json