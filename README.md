## Dependencies

Maven - install this first, the command line tool

## To Run

```bash
cd gridworld
make
make small # this will run small grid world
make large # this will run large grid world
```

You should be able to open the project up in IntelliJ. I've been running it from there,
but you'll likely still need maven.


Since I'm writing this very last-second, I will not be able to offer any support by the deadline of the project. Sorry! Good luck though!

## Your Results
So, if you want to track how many iterations of VI and PI the code is doing, just look at the terminal for the following lines:

Passes: 6
...
Finished VI: ...
__ 'Passes' is the number of VI iterations __

Total policy iterations: 4
...
Finished PI: ...
__ Total PI iterations are given here __

I'm mostly certain about these results being the right iteration numbers, but since I'm not 100% familiar with BURLAP I could be wrong. 

Q-Learning will have plots generated for you automatically at the end. If it doesn't work or if it shows up very tiny, I'm afraid I haven't found a solution for you. 

