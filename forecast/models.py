from django.db import models

class UserInfo(models.Model):
    name = models.CharField(max_length=100)
    place = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=15)

    def __str__(self):
        return self.name
