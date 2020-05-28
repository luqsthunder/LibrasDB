import xml.etree.ElementTree as ET

tree = ET.parse('Inventário+Libras/FLN G1 D1 CONVER Copa2014 v323/sub0.xml')
tiers = tree.findall('TIER')
print(tiers, len(tiers))
print(tiers[0].findall('ANNOTATION'))
