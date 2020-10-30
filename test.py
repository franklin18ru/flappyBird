import itertools

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
first = ( list(split(range(512), 15)) )
second = range(-8, 11)  
third = ( list(split(range(309), 15)) )
fourth = ( list(split(range(100,512), 15)))

for i in range(309):
    print(i)

newList = [first, second, third, fourth]
newList = list(itertools.product(*newList))

state = "18.0,0.0,69.0,119"
state = state.split(",")

qvalues = []


for i in range(len(newList)):
    if float(state[0]) in newList[i][0] and float(state[1]) == newList[i][1] and float(state[2]) in newList[i][2] and float(state[3]) in newList[i][3]:
        print("state[0]", state)
        print("range", newList[i])

# print( list(split(range(512), 4)) )  "player_y"
# print( list(split(range(309), 4)) ) "next_pipe_dist_to_player" 
# print( list(split(range(100,512), 4))) next_pipe_top_y"

#   1: "player_y"
#   2: "player_vel" 
#   3: "next_pipe_dist_to_player" 
#   4: "next_pipe_top_y"


#if float(state[0]) in newList[i][0] and state[1] == i[1] and float(state[2]) in i[2] and int(state[3]) in i[3]: