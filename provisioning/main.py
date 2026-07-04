import harmony

from harmony.core.util import App

if __name__ == "__main__":
    apps_list = [App("app1", .2, 5), App("app2", .3, 2), App("app3", .5, 3),App("app4", 3, 1)]
    groups, cost = harmony.Algorithm(apps_list)
    print("Provisioning plan:")
    for i in range(len(groups)):
        print("The configurations of the group " + str(i) + " is: ", end='')
        print(groups[i], end="----\n")
    print("The cost of provisioning plan is:", cost)
