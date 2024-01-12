// MySQLConnectionPool.h
#ifndef MYSQL_CONNECTION_POOL_H
#define MYSQL_CONNECTION_POOL_H

// #include <cppconn/connection.h>
#include <cppconn/connection.h>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>


#include <mutex>
#include <queue>
#include <string>
#include <memory>
#include "Logger.h"

class MySQLConnectionPool : public Logger {
public:
    MySQLConnectionPool(const std::string& host, const std::string& user, const std::string& pass, const std::string& db, size_t maxConn = 10);
    std::shared_ptr<sql::Connection> getConnection();
    void releaseConnection(std::shared_ptr<sql::Connection> conn);

private:
    std::string host_, user_, pass_, db_;
    size_t maxConn_;

    std::shared_ptr<sql::Connection> createConnection();

    std::mutex pool_mutex_;
    std::queue<std::shared_ptr<sql::Connection>> conn_queue_;
    // size_t maxConn_;
    size_t currentConn_ = 0;
    // std::string host_, user_, pass_, db_;
};

#endif // MYSQL_CONNECTION_POOL_H
