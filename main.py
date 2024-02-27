import harmony

from harmony.core.util import App

if __name__ == "__main__":
    apps_list = [App("app0", 0.5, 5), App("app1", 0.6, 5), App("app2", 0.8, 5), App("app3", 1, 15)]
    groups, cost = harmony.Algorithm(apps_list)
    print("Provisioning plan:")
    for i in range(len(groups)):
        print("The configurations of the group " + str(i) + " is: ", end='')
        print(groups[i], end="----\n")
    print("The cost of provisioning plan is:", cost)