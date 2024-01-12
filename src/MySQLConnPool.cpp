// #include <cppconn/driver.h>
// #include <cppconn/exception.h>
// #include <cppconn/connection.h>
// #include <cppconn/statement.h>  // 确保包含了这个头文件
// #include <cppconn/resultset.h>
// #include <mutex>
// #include <queue>
// #include <memory>
// #include <iostream>

// MySQLConnectionPool.cpp
#include "MySQLConnPool.h"
#include <cppconn/driver.h>
#include <cstddef>
#include <stdexcept>

MySQLConnectionPool::MySQLConnectionPool(const std::string& host, const std::string& user, const std::string& pass, const std::string& db, size_t maxConn)
    : host_(host), user_(user), pass_(pass), db_(db), maxConn_(maxConn), currentConn_(0) {
    for (size_t i = 0; i < maxConn_; ++i) {
        auto conn = createConnection();
        if (!conn) {
            LOG_FATAL("create connection failed");
            // LOG_FATAL("getConnection failed");
            exit(-1);
        } 
        conn_queue_.push(conn);
    }
}

std::shared_ptr<sql::Connection> MySQLConnectionPool::getConnection() {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    if (conn_queue_.empty()) {
        if (currentConn_ < maxConn_) {
            auto newConn = createConnection();
            if (newConn) {
                conn_queue_.push(newConn);
                ++currentConn_;
            } else {
                LOG_ERROR("getConnection failed");
                return nullptr; // 返回空指针
            }
        } else {
            LOG_ERROR("getConnection greater than maxConn");
            return nullptr; // 返回空指针
        }
    }
    auto conn = conn_queue_.front();
    conn_queue_.pop();
    LOG_INFO("getConnection success");
    return conn;
}
// std::shared_ptr<sql::Connection> MySQLConnectionPool::getConnection() {
//     std::unique_lock<std::mutex> lock(pool_mutex_);
//     if (conn_queue_.empty()) {
//         if (currentConn_ < maxConn_) {
//             conn_queue_.push(createConnection());
//             ++currentConn_;
//         } else {
//             throw std::runtime_error("Reached maximum number of connections");
//         }
//     }
//     auto conn = conn_queue_.front();
//     conn_queue_.pop();
//     return conn;
// }

void MySQLConnectionPool::releaseConnection(std::shared_ptr<sql::Connection> conn) {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    conn_queue_.push(conn);
    LOG_INFO("releaseConnection success");
}

// std::shared_ptr<sql::Connection> MySQLConnectionPool::createConnection() {
//     sql::Driver* driver = get_driver_instance();
//     std::shared_ptr<sql::Connection> conn(driver->connect(host_, user_, pass_));
//     conn->setSchema(db_);
//     return conn;
// }
std::shared_ptr<sql::Connection> MySQLConnectionPool::createConnection() {
    try {
        sql::Driver* driver = get_driver_instance();
        std::shared_ptr<sql::Connection> conn(driver->connect(host_, user_, pass_));
        conn->setSchema(db_);
        return conn;
    } catch (sql::SQLException &e) {
        std::cerr << "SQLException in createConnection: " << e.what() << std::endl;
        return nullptr; // 返回空指针
    }
}


// class MySQLConnectionPool {
// public:
//     MySQLConnectionPool(const std::string& host, const std::string& user, const std::string& pass, const std::string& db, int maxConn = 10) 
//         : host_(host), user_(user), pass_(pass), db_(db), maxConn_(maxConn) {
//         for (int i = 0; i < maxConn_; ++i) {
//             conn_queue_.push(createConnection());
//         }
//     }

//     std::shared_ptr<sql::Connection> getConnection() {
//         std::unique_lock<std::mutex> lock(pool_mutex_);
//         if (conn_queue_.empty()) {
//             if (currentConn_ < maxConn_) {
//                 conn_queue_.push(createConnection());
//                 ++currentConn_;
//             } else {
//                 throw std::runtime_error("Reached maximum number of connections");
//             }
//         }
//         auto conn = conn_queue_.front();
//         conn_queue_.pop();
//         return conn;
//     }

//     void releaseConnection(std::shared_ptr<sql::Connection> conn) {
//         std::unique_lock<std::mutex> lock(pool_mutex_);
//         conn_queue_.push(conn);
//     }

// private:
//     std::shared_ptr<sql::Connection> createConnection() {
//         sql::Driver* driver = get_driver_instance();
//         std::shared_ptr<sql::Connection> conn(driver->connect(host_, user_, pass_));
//         conn->setSchema(db_);
//         return conn;
//     }

//     std::mutex pool_mutex_;
//     std::queue<std::shared_ptr<sql::Connection>> conn_queue_;
//     int maxConn_;
//     int currentConn_ = 0;
//     std::string host_, user_, pass_, db_;
// };

// int main() {
//     try {
//         MySQLConnectionPool pool("localhost:3307", "ai", "12345678", "mydatabase", 5);

//         auto conn = pool.getConnection();
//         std::shared_ptr<sql::Statement> stmt(conn->createStatement());
//         // stmt->execute("SELECT * FROM users");

//         // 插入数据
//         stmt->executeUpdate("INSERT INTO users (name, age) VALUES ('John Doe', 30)");

//         // 执行查询
//         std::shared_ptr<sql::ResultSet> res(stmt->executeQuery("SELECT * FROM users"));

//         // 遍历结果集
//         while (res->next()) {
//             // 获取每一行的数据
//             std::cout << "ID: " << res->getInt("id");
//             std::cout << ", Name: " << res->getString("name");
//             std::cout << ", Age: " << res->getInt("age") << std::endl;
//         }
        
//         // 删除数据
//         stmt->executeUpdate("DELETE FROM users WHERE name = 'John Doe'");

//         pool.releaseConnection(conn);
//     } catch (sql::SQLException &e) {
//         std::cerr << "SQLException in " << __FILE__;
//         std::cerr << " (" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
//         std::cerr << "Error: " << e.what();
//         std::cerr << " (MySQL error code: " << e.getErrorCode();
//         std::cerr << ", SQLState: " << e.getSQLState() << " )" << std::endl;
//     } catch (std::exception &e) {
//         std::cerr << "Exception: " << e.what() << std::endl;
//     }

//     return 0;
// }
