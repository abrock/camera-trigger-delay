
Sensor readout speed measurements:

Sony Alpha 1, electronic shutter:
Readout time: 3.704ms +- 1.5us, 50 Samples, range: [3.700, 3.706]
Readout speed stats: 1561.7 +- 0.6 lines per ms, 50 Samples, range: [1560.9, 1563.1]

Sony Alpha 7R II, electronic shutter:
Readout time stats: 75.028ms +- 0.58us, 50 Samples, range: [75.027, 75.029]
Readout speed stats: 70.907 +- 5.5e-04 lines per ms, 50 Samples, range: [70.906, 70.908]

Sony Alpha 7R IV, electronic shutter:
Readout time stats: 95.791ms +- 0.15us, 50 Samples, range: [95.791, 95.791]
Readout speed stats: 66.561 +- 1.0e-04, lines per ms 50 Samples, range: [66.561, 66.562]

Sony Alpha 7R IV, mechanical shutter with electronic front curtain:
Readout time stats: 2.786ms +- 14us, 50 Samples, range: [2.749, 2.821]
Readout speed stats: 2288.42 +- 11.5, 50 Samples, range: [2260.15, 2319.32]

Sony RX10 IV, electronic shutter:
Readout time stats: 7.439ms +- 1.9us, 60 Samples, range: [7.435, 7.444]
Readout speed stats: 493.60 +- 0.13, 60 Samples, range: [493.28, 493.90]

Sony RX10 IV, 1000fps HFR mode:
Readout time stats: 0.8526ms +- 13.1us, 60 Samples, range: [0.8349, 0.8797]
Readout speed stats: 1267.02 +- 19.25, 60 Samples, range: [1227.73, 1293.63]
Readout time relative to photo mode: 11.5% (~1/8)

Sony RX10 IV, 500fps HFR mode:
Readout time stats: 1.674ms +- 5.6us, 60 Samples, range: [1.666, 1.684]
Readout speed stats: 645.15 +- 2.2, 60 Samples, range: [641.32, 648.38]
Readout time relative to photo mode: 22.5% (~1/4)

Sony RX10 IV, 250fps HFR mode:
Readout time stats: 3.484ms +- 1.7us, 60 Samples, range: [3.477, 3.487]
Readout speed stats: 309.95 +- 0.15, 60 Samples, range: [309.74, 310.66]
Readout time relative to photo mode: 46.8% (~1/2)


Trigger delay measurements:

For Sony A7R II with electronic shutter:
1. Trigger
2. `_delay_ms(150)`
3. Run the following color sequence with 50ms delay between each color:

RGB
100 red
110 yellow
111 white


For Sony A1 with electronic shutter:
1. Trigger
2. `_delay_ms(73)`
3. Run the following color sequence with 3ms delay between each color:

RGB
100 red
110 yellow
111 white
101 pink
001 blue
011 turqoise
010 green


For Sony A7R IV with mechanical shutter, EFCS:
1. Trigger
2. `_delay_ms(60)`
3. Run the color sequence with 3ms delay between each color:
