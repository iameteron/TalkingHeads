### ls20:

Level-1

Hardcoded actions:
```
Just execute actions in this order: ACTION3, ACTION3, ACTION3, ACTION1, ACTION1, ACTION1, ACTION1, ACTION4, ACTION4, ACTION4, ACTION1, ACTION1, ACTION1
```

Prefered answers:
```
Main goal is to navigate blue-orange square to the end of the maze after making shape presented in the left-down corner and shape at the end of the maze match. Shape in the left-down corner rotates by 90 degrees when stepping on white cross symbol located somewhere on a map. After stepping on cross ask me question if the shapes are matching so I can confirm.
```

```
You control orange-blue square and can use actions to navigate: ACTION1 - UP, ACTION2 - DOWN, ACTION3 - LEFT, ACTION4 - RIGHT.
```

Level-2

```
On the level two mechanics are mostly the same: match shapes using white cross and then navigate to the shape in the end of the maze. This level introduces one new mechanic - yellow circle which refills your yellow timer bar. You need to use it when your yellow status bar gets low, ask me when to use it when you about to do it.
```

### ar25:

Prefered answers:

```
Your goal is to move the grey shape to match with the yellow shape, to do this use your navigation actions: ACTION1 - UP, ACTION2 - DOWN, ACTION3 - LEFT, ACTION4 - RIGHT.
```

```
In this level you move black and white shape with your navigation keys, grey shape moves match the movement when doing up and down action, but moves the opposite when applying left or right actions.
```

### bp35:

Prefered answers:

```
You control the blue-yellow model, your goal is to navigate to the end of the maze. You can navigate left and right, your gravity is reversed so when you got no blocks above your model you will "fall up", you cannot die and you need to fall up to progress in the maze.
```

```
You can use ACTION3 - LEFT, ACTION4 - RIGHT. And also you can destroy green blocks with ACTION6 + "coords of the green block" to progress further. To determine coords of a given block that you want to destroy, remember that upper-left coordinate of a frame is [0, 0], and bottom-right coordinate is [63, 63].
```

### lp85:

Prefered answers:

Level-1
```
You need to rotate a loop of tiles so the yellow tile goes to yellow tile end zone that is indicated by 4 yellow dots. 
To rotate a loop of tiles clockwise you can use Green Button by executing ACTION6 56 29.
To rotate a loop of tiles counter-clockwise you can use Red Button by executing ACTION6 4 29.
```

Level-2
```
You need to rotate loops of tiles so the yellow tiles goes to yellow tiles end zone that is indicated by 4 yellow dots.

You can rotate 3 loops:

To rotate loop of tiles 1 clockwise you can use Green Button by executing ACTION6 20 16.
To rotate loop of tiles 1 counter-clockwise you can use Red Button by executing ACTION6 38 16.

To rotate loop of tiles 2 clockwise you can use Green Button by executing ACTION6 14 25.
To rotate loop of tiles 2 counter-clockwise you can use Red Button by executing ACTION6 47 25.

To rotate loop of tiles 3 clockwise you can use Green Button by executing ACTION6 14 34.
To rotate loop of tiles 3 counter-clockwise you can use Red Button by executing ACTION6 47 34.
```

