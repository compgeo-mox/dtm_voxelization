Subject: Rialba source points sit about 40 m above the terrain model


We placed the seven surveyed source points from rialba_source_pts.txt on our
Rialba terrain model. Horizontally they fall exactly where we expect them, but
every one of them comes out around 40 m higher than the ground beneath it. The
offset is far too consistent to be measurement noise, so the survey and the
terrain model are probably not on the same reference, and we would like your
help deciding which of the two is the odd one.

What we compared: the Est, Nord and Quota columns of the survey, read as
UTM32N, against the elevation of our DTM point cloud interpolated at each
point's Est and Nord. The elevations below are absolute metres.

Point 107, Est 527571.009, Nord 5082129.486: survey 274.513, terrain 233.9,
difference +40.6 m
Point 104, Est 527579.693, Nord 5082124.468: survey 275.374, terrain 234.7,
difference +40.6 m
Point 105, Est 527586.802, Nord 5082117.228: survey 275.716, terrain 235.6,
difference +40.1 m
Point 109, Est 527595.628, Nord 5082112.984: survey 275.762, terrain 236.6,
difference +39.1 m
Point 108, Est 527551.913, Nord 5082152.059: survey 272.181, terrain 231.5,
difference +40.7 m
Point 106, Est 527558.192, Nord 5082144.039: survey 273.200, terrain 232.0,
difference +41.2 m
Point 110, Est 527564.015, Nord 5082136.179: survey 274.249, terrain 232.4,
difference +41.9 m

The differences average +40.6 m and range from +39.1 to +41.9, with a standard
deviation of 0.79 m.

The ground there is a slope of about 29 degrees, 41 at its steepest, rising
43.5 m across a 60 by 60 m box around the points. It is not a cliff, so this
is not a case of an interpolated elevation being meaningless: at these
coordinates our model is well behaved and the survey simply sits above it.

Two explanations fit the size and the sign of the difference.

- The heights are ellipsoidal rather than orthometric. A GNSS survey whose
  heights were never reduced to sea level sits above an orthometric terrain
  model by the geoid undulation, which around here is roughly 48 m. That is a
  few metres more than what we see, but the sign and the order of magnitude
  match.

- The offset is horizontal rather than vertical. On a 29 degree slope, 40 m of
  elevation corresponds to about 70 m on the ground. A survey delivered in a
  different datum, Roma 40 / Gauss-Boaga instead of ETRS89 / UTM32N for
  example, would be displaced by roughly that much, and would then read to us
  as a height error.

Telling the two apart needs the survey's own metadata rather than more
computation on our side. Could you confirm which datum the coordinates are in,
and whether the Quota column is ellipsoidal or orthometric, and if orthometric
which geoid model was used to reduce it?

In the meantime we are using the coordinates exactly as delivered, with no
vertical correction applied, so nothing downstream depends on the answer yet.
