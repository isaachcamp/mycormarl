# Fine-root trait defaults for cultivated *Daucus carota*

**Research date:** 12 August 2026

## Decision

The default plant traits represent a late, deep-sand greenhouse carrot
regime—not generic field or wild *Daucus carota*. They are:

| Runtime field | Default | Meaning |
|---|---:|---|
| `kfroot` | `0.18` | Fine/fibrous-root DM / whole-plant DM |
| `specific_root_length` | `254.343 m g-1` | Fine-root length / fine-root DM |
| `root_radius` | `0.01 cm` (`0.10 mm`) | Effective fine-root absorbing radius |

The model reconstructs represented fine-root length as

```text
plant_DM × kfroot × SRL
```

`kfroot` is the model's only active plant-absorber fraction. Every centimetre
of the resulting represented fine-root length is uptake-active; the model does
not apply a further inactive-tissue, coarse-root, or active-absorber fraction.
This is a modelling assumption, not a claim that every measured fibrous root is
physiologically active in a cultivated carrot.

The default lies within the observed silica-sand uncertainty interval
`0.119–0.244`. The SRL and radius are compatible fine-root-system aggregates
but are not paired with those mass measurements.

## Matched mass-fraction evidence and scope

Westerveld (2005, Table 2.18 and Appendix Table A2.16) separately measured
top, storage-root, and fibrous-root dry mass for six-month `Idaho` and
`Fontana` carrots grown one per 10-cm-diameter, 150-cm-deep PVC column of 98%
silica sand. The required denominator is directly observed:

```text
kfroot = fibrous_root_DM / (top_DM + storage_root_DM + fibrous_root_DM)
```

| Quantity | Range across six treatment means | Median default |
|---|---:|---:|
| Fine/fibrous-root / whole-plant DM (`kfroot`) | `0.119–0.244` | `0.18` runtime representative value |

This is an exploratory controlled regime: 35.2–61.4% of fibrous-root mass was
below 30 cm and roots reached the 150-cm column maximum. Use its observed row
pairs, rather than independent extrema, for sensitivity within that regime.

It is not a field-carrot default. In mature field-grown `Nantes Duke`, Pietola
(1995) measured `0.0177–0.0329` fibrous-root / total-root DM in four fine-sand
treatments. It is a different denominator, so it cannot set the lower endpoint
of `kfroot` without matched shoot and storage-root dry masses. No separated
fine-root dry-mass observation was identified for wild carrot.

## Fine-root radius and SRL evidence

The default radius is supported as an **effective fine-root-system radius**.
The GRooT *D. carota* aggregate has a median fine-root diameter of
`0.2036016475 mm` (12 study-site entries; IQR `0.17314191725–0.2285 mm`). Its
half, `0.01018 cm`, rounds to `0.01 cm`. All 16 underlying mean-diameter
records are classified as fine root (`FR`).

The same GRooT release supplies the default SRL median, `254.34303615 m g-1`
(ten study-site entries); all contributing records are `FR`, with two explicit
0–2-mm diameter classes and the remainder without a reported diameter or root
order. Thus neither trait describes every root segment or a single branch order.
The aggregates have separate sites and plants, so they are not a matched
diameter–SRL dataset.

As a direct cultivated-field check, Pietola and Smucker (1998) report that
75–90% of fibrous-root length was in approximately `0.15-mm` roots and SRL was
`250–350 m g-1`. This brackets the default SRL and implies a `0.0075-cm` radius.
Use `0.0075–0.0114 cm` for radius sensitivity. The current smooth-cylinder
pair gives `1,598 cm2 g-1` lateral area, consistent in scale with their
reported `1,500–2,000 cm2 g-1` fibrous specific surface area; it is only a
geometric cross-check, not an identity of measurements.

## Evidence limits

- `kfroot` is a standing dry-mass observation, not a marginal allocation rule,
  tissue-activity fraction, or root-length diameter fraction.
- The deep-sand whole-plant observation, GRooT species aggregates, and field fibrous-root
  traits are different samples and regimes. The defaults make that scope
  explicit; they do not assert a universal carrot phenotype.
- The field mass-fraction range is not yet paired to a verified field
  total-root / whole-plant denominator.

## Sources

1. Westerveld, S. M. (2005). *Nitrogen dynamics of the carrot crop and
   influences on yield and Alternaria and Cercospora leaf blights*, pp. 37–38,
   68–70. [Full thesis](https://bradford-crops.uoguelph.ca/sites/default/files/Sean%20Westerveld%20Thesis.pdf).
2. Pietola, L. (1995). [Effect of soil compactness on the growth and quality
   of carrot](https://doi.org/10.23986/afsci.72611).
3. Pietola, L. & Smucker, A. J. M. (1998). [Fibrous carrot root responses to
   irrigation and compaction](https://doi.org/10.1023/A:1004294330427).
4. Guerrero-Ramírez, N. R. et al. (2021). [Global root traits (GRooT)
   database](https://doi.org/10.1111/geb.13179).
