#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <map>
#include <boost/algorithm/string.hpp>

using namespace std;

class NaiveB
{
    private:
        struct metadata
        {
            int column = NULL;
            int overall_freq = 0;
            double overall_prob = 0;
        };   

        struct likelihood
        {
            int y_freq = 0;
            double y_prob = 0;
            double likelihood = 0;
        };

        vector <vector<string> > data;

        map <string, metadata*> Features;
        map <string, metadata*> Y;

        map <string, map<string, likelihood*> > connections;
        map <string, double> result;
      
        int data_size;
        int num_class;

        void getFeatures()
        {
            for (auto &row : data)
            {
                for (int col = 0; col < row.size()-1; col ++)
                {
                    if (Features.find(row[col]) == Features.end())
                    {
                        metadata* new_entryX = new metadata;
                        map<string, likelihood*> edge;

                        new_entryX->overall_freq += 1;
                        new_entryX->column = col;
 
                        Features[row[col]] = new_entryX;
                        connections[row[col]] = edge;
                        cout << "New X: " << row[col] << endl;
                    }
                    else
                    {
                        Features[row[col]]->overall_freq += 1;
                    }
                }

                if (Y.find(row[row.size()-1]) == Y.end())
                {
                    metadata* new_entryY = new metadata;
                    new_entryY->overall_freq += 1;
                    Y[row[row.size()-1]] = new_entryY;
                    cout << "New Y: " << row[row.size()-1] << endl;
                }
                else
                {
                    Y[row[row.size()-1]]->overall_freq += 1;
                }
            }
        }

        void updateGProb()
        {
            //updating individual X global Probability
            for (auto &feature : Features)
            {
                feature.second->overall_prob = double(feature.second->overall_freq) / double(data_size);
            }

            //update individual Y global Probability
            for (auto &y : Y)
            {
                y.second->overall_prob = double(y.second->overall_freq) / double(data_size);
            }

        }

        void updateC()
        {
            int pass = 0;
            //update connection between Feature and Y
            for (auto &con : connections)
            {
                for (auto &y : Y )
                {
                    for (auto &row : data)
                    {
                        if (y.first == row[row.size()-1] && row[Features[con.first]->column] == con.first)
                        {
                            if (con.second.find(y.first) == con.second.end())
                            {
                                likelihood* item = new likelihood();
                                item->y_freq += 1;
                                con.second[y.first] = item;
                            }
                            else
                            {
                                con.second[y.first]->y_freq += 1;
                            }
                            pass += 1;
                        }
                    }
                    cout << "pass : " << pass << endl;

                    // update connection probability
                    con.second[y.first]->y_prob = double(con.second[y.first]->y_freq) / double(Features[con.first]->overall_freq);
                    cout << con.first << " " << y.first << " prob: " << con.second[y.first]->y_prob << endl;
                }
            } 
        }
        
        
        void insert(const vector<string> &item)
        {
            data.push_back(item);
        }

        void load(string filename)
        {
            ifstream file(filename);
            string line;

            while (getline(file, line))
            {
                vector<string> item;
                istringstream input(line);
                for (string temp; input >> temp;)
                    item.push_back(temp);
                insert(item);
            }
        }   

       /*
        void updateLTable()
        {
            for (auto &entry : likelihood_table)
            {
                for (auto &y : test_feature)
                {
                    for (auto &row : data)
                    {
                        if (row[entry.first] == "1" && row[row.size()-1] == y.first)
                        {
                            entry.second[y.first]->yes_count += 1;
                        }
                    }
                    entry.second[y.first]->yes_prob = double(entry.second[y.first]->yes_count) / double(classes[entry.first]->freq);
                }
            }
        }

        double totalXProb() const
        {
            double total_prob = 0.0;
            for (auto &feature : classes)
            {
                if (total_prob == 0.0)
                {
                    total_prob = feature.second->prob;
                }
                else
                {
                    total_prob *= feature.second->prob;
                }
            }
            
            return total_prob;

        }

        void getResult()
        {
            double productXProb = totalXProb();
            
            for (auto &y : test_feature)
            {
                double top_equation  = 0.0;

                for (auto &current : likelihood_table)
                {
                    if (top_equation == 0.0)
                    {
                        top_equation = current.second[y.first]->yes_prob;
                    }
                    else
                    {
                        top_equation *= current.second[y.first]->yes_prob;
                    }

                }

                top_equation *= y.second->prob;
                y.second->chance = double(top_equation / productXProb);                 
            }

        }
        void setLTable()
        {
            
            for (auto &entry : classes)
            {
                map<string, likelihood*> y_map;
                map<string, metadata*> current;
                for (auto &key : test_feature )
                {
                   
                    likelihood* item = new likelihood();
                    y_map[key.first] = item;

                } 
                //grab columns aka features 
                likelihood_table[entry.first] = y_map;                   
            }

        }
    
    */

    public:     
        NaiveB(string filename, int num_classes) 
        {
            load(filename);
            num_class = num_classes;
            data_size = data.size();
            getFeatures(); updateGProb(); updateC();
        }

        void displayD()
        {
            for (auto &entry: data)
            {
                for (auto &item : entry)
                {
                    cout << item << " ";
                }
                cout << "\n";
            }
        }

        void displayT()
        {
            cout << "Displaying X " << endl;
            for (auto &feature: Features)
            {
                cout << "Feature: " << feature.first << " occurences: " << feature.second->overall_freq << " prob: " << feature.second->overall_prob << endl;
            }

            cout << "\nDisplaying Y " << endl;
            for (auto &y: Y)
            {
                cout << "Y: " << y.first << " occurences: " << y.second->overall_freq << " prob: " << y.second->overall_prob << endl;
            }

            cout << "\nDisplay C Table " << endl;
            for (auto &con : connections)
            {
                for (auto &edge : con.second)
                {
                    cout << "Feature: " << con.first << " to Y: " << edge.first << " ocurrences: " << edge.second->y_freq << " prob: " << edge.second->y_prob << endl;
                }
            }
        }
     
};

int main()
{
    NaiveB classifier = NaiveB("dataset.txt", 3);
    classifier.displayT();
}