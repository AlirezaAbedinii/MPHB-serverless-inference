import harmony

from harmony.core.util import App

if __name__ == "__main__":
    apps_list = [App("app1", 6.0, 0.1), App("app2", 4.5, 0.2), App("app3", 5.5, 0.15),App("app4", 3, 0.3)]
    groups, cost = harmony.Algorithm(apps_list)
    print("Provisioning plan:")
    for i in range(len(groups)):
        print("The configurations of the group " + str(i) + " is: ", end='')
        print(groups[i], end="----\n")
    print("The cost of provisioning plan is:", cost)
