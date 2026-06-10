> Clock speed and Processors performance are not a same thing 

```
SPECint95	SPECfp95
195 MHz	MIPS R10000	11.0	17.0
400 MHz	Alpha 21164	12.3	17.2
300 MHz	UltraSPARC	12.1	15.5
300 MHz	Pentium II	11.6	8.8
300 MHz	PowerPC G3	14.8	11.4
135 MHz	POWER2	6.2	17.6

```
## Pipelining & Instruction level parallelism 
conventional thinking tells Instructions are excuted one afer another but that's not really what happens in fact post 1980s several instructions are all partially executing at the same time

```
Sequential:
I1: F D E W
I2:         F D E W
I3:                 F D E W

Pipelined:
I1: F D E W
I2:   F D E W
I3:     F D E W
```
with pipelined sequence CPI gets four fold speed withour chnaging clock speed by completing 1 istruction per cycle

At the beginning of each cycle the data and control information are partially processed instruction is held in a pipeline patch




# notes https://www.lighterra.com/papers/modernmicroprocessors/