// Canonical NER label set for ClimaTag — mirror of ner_labels.py at the repo root.
// The 28 CliReNER categories + the custom Climate Model category.
// Keep in sync with ner_labels.py.
export const NER_LABELS = [
  'Asset', 'Body Part', 'Body of Water', 'Chemical', 'Disease',
  'Ecosystem', 'Energy Source', 'Field of Study', 'Geographical Feature',
  'Intellectual Artefact', 'Location', 'Mathematical Expression',
  'Measuring Device', 'Meteorological Phenomenon', 'Method',
  'Natural Disaster', 'Natural Phenomenon', 'Organism', 'Organization',
  'Other', 'Person', 'Physical Artefact', 'Physical Phenomenon',
  'Policy', 'Quantity', 'Satellite', 'System', 'Time Period',
  'Climate Model',
]